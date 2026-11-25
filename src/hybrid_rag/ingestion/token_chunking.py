"""Optional refinement of existing Chunks; no new persisted data hierarchy.

Only over-budget embedding_text is touched. All emitted text is assembled from
existing source slices, including repeated fences, headers and overlap context.
"""
import ast
import hashlib
import json
import re
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from markdown_it import MarkdownIt

from hybrid_rag.schemas.document import ChunkPart, stable_hash
from .provenance import slice_spans

TOKEN_CHUNKER_VERSION = "token-refinement-v2-hard-limit"


@lru_cache(maxsize=8)
def _tokenizer_signature(tokenizer):
    # The selected tokenizer configuration is fixed throughout one run.
    backend=json.loads(tokenizer.backend_tokenizer.to_str())
    for key in ('truncation','padding'):backend.pop(key,None)
    return stable_hash([backend,tokenizer.special_tokens_map])


class TokenRefiner:
    def __init__(self, tokenizer, *, target_tokens=384, max_tokens=512, overlap=32, chunk_size=800, input_prefix=''):
        if not 0 <= overlap < target_tokens <= max_tokens:
            raise ValueError("Require 0 <= overlap < target_tokens <= max_tokens")
        self.tokenizer = tokenizer
        self.target, self.maximum, self.overlap = target_tokens, max_tokens, overlap
        self.chunk_size = chunk_size  # existing oversized flag remains character-based
        self.input_prefix=input_prefix
        self.config = dict(target_tokens=target_tokens, max_tokens=max_tokens, overlap=overlap,input_prefix=input_prefix)
        self.fingerprint = _tokenizer_signature(tokenizer)
        self.findings = []
        self.changes = []
        self.reconstructions = []

    @lru_cache(maxsize=32768)
    def count(self, text, special=True):
        return len(self.tokenizer.encode(text, add_special_tokens=special, truncation=False))

    @staticmethod
    def _text(text, ranges):
        return ''.join(text[a:b] for a, b, _ in ranges)

    def _pack(self, text, units, prefix, head=(), tail=()):
        """Pack whole units; repeat whole units only within the overlap budget."""
        groups, current = [], []
        def render(group):return [*head, *[(a,b,'body') for a,b in group], *tail]
        for unit in units:
            candidate = [*current, unit]
            if current and self.count(prefix+self._text(text, render(candidate))) > self.target:
                groups.append(current); current = []
            current.append(unit)
        if current:groups.append(current)
        result = []
        for i, group in enumerate(groups):
            overlap = []
            if i and self.overlap:
                for a,b in reversed(groups[i-1]):
                    proposal = [(a,b,'context'), *overlap]
                    if self.count(self._text(text,proposal), False) > self.overlap:break
                    if self.count(prefix+self._text(text,[*head,*proposal,*render(group)[len(head):]])) > self.maximum:break
                    overlap = proposal
            # A repeated table header is context; code fences are wrappers.
            h = [(a,b,'context' if role=='body' and i else role) for a,b,role in head]
            result.append([*h,*overlap,*[(a,b,'body') for a,b in group],*tail])
        return result

    def _plain(self, text, a, b, prefix):
        budget = self.target-self.count(prefix)
        if budget <= self.overlap:
            budget = self.maximum-self.count(prefix)
        if budget<=0:return [("unresolved_heading_budget",[(a,b,'body')])]
        overlap=min(self.overlap,budget-1)
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer, chunk_size=budget, chunk_overlap=overlap,
            strip_whitespace=False, keep_separator=True)
        source=text[a:b]
        pieces=splitter.split_text(source)
        if not pieces or any(not p.strip() for p in pieces):return [('unresolved_text_alignment',[(a,b,'body')])]
        # Token overlap is not a character offset. Repeated text may admit
        # several exact matches; find an ordered, gap-free alignment whose last
        # piece reaches the original end, rather than guessing one occurrence.
        remaining=[0]*(len(pieces)+1)
        for i in range(len(pieces)-1,-1,-1):remaining[i]=remaining[i+1]+len(pieces[i])
        @lru_cache(maxsize=None)
        def align(i,previous_end):
            if i==len(pieces):return () if previous_end==len(source) else None
            if previous_end+remaining[i]<len(source):return None
            floor=previous_end-len(pieces[i-1])+1 if i else 0
            position=source.find(pieces[i],floor)
            while 0<=position<=previous_end:
                end=position+len(pieces[i])
                if end>previous_end and self.count(source[position:previous_end],False)<=overlap:
                    rest=align(i+1,end)
                    if rest is not None:return ((position,end),*rest)
                position=source.find(pieces[i],position+1)
            return None
        try:ranges=align(0,0)
        except RecursionError:ranges=None
        if ranges is None:return [('unresolved_text_alignment',[(a,b,'body')])]
        groups=[('text_recursive',[(a+x,a+y,'body')]) for x,y in ranges]
        if any(self.count(prefix+self._text(text,r))>self.maximum for _,r in groups):
            return [("unresolved_text_budget",[(a,b,'body')])]
        return groups

    def _plan(self, text, prefix):
        parsed = MarkdownIt().enable('table').parse(text)
        offsets = [0]
        for line in text.splitlines(True):offsets.append(offsets[-1]+len(line))
        top = [t for t in parsed if t.level==0 and t.map and t.nesting!=-1]
        if not top:return self._plain(text,0,len(text),prefix)
        blocks = [(0 if i==0 else offsets[t.map[0]], offsets[top[i+1].map[0]] if i+1<len(top) else len(text),t)
                  for i,t in enumerate(top)]
        output, pending = [], []
        def flush():
            output.extend(('markdown_blocks',r) for r in self._pack(text,pending,prefix))
            pending.clear()
        for a,b,token in blocks:
            if self.count(prefix+text[a:b])<=self.maximum:
                pending.append((a,b)); continue
            flush()
            if token.type=='fence':
                begin,end = offsets[token.map[0]+1],offsets[token.map[1]-1]
                body = text[begin:end]
                closing = text[end:offsets[token.map[1]]].strip()
                if body!=token.content or not re.fullmatch(re.escape(token.markup[0])+'{'+str(len(token.markup))+',}',closing):
                    output.append(('unresolved_fence_layout',[(a,b,'body')])); continue
                language = token.info.split()[0].lower() if token.info.split() else ''
                cuts, method = [], 'unresolved_unsafe_fence'
                if language in {'python','py'}:
                    try:
                        tree = ast.parse(body)
                        lines=[0]
                        for line in body.splitlines(True):lines.append(lines[-1]+len(line))
                        cuts=[0]+[lines[min([n.lineno,*[d.lineno for d in getattr(n,'decorator_list',[])]])-1] for n in tree.body[1:]]+[len(body)]
                        for x,y in zip(cuts,cuts[1:]):ast.parse(body[x:y])
                        method='python_ast'
                    except (SyntaxError,ValueError,RecursionError):cuts=[]
                if not cuts and language in {'','result','output','text','log','logs'}:
                    events=list(re.finditer(r'(?m)^(?:={3,}[^\n]*\bMessage\b[^\n]*={3,}|\[?\d{4}-\d{2}-\d{2}[T ][0-9:.]+[^\n]*)\n?',body))
                    if len(events)>1:
                        cuts=[0]+[m.start() for m in events[1:]]+[len(body)];method='log_events'
                if len(cuts)<3:
                    output.append(('unresolved_indivisible_code',[(a,b,'body')])); continue
                units=[(begin+x,begin+y) for x,y in zip(cuts,cuts[1:])]
                output.extend((method,r) for r in self._pack(text,units,prefix,[(a,begin,'wrapper')],[(end,b,'wrapper')]))
            elif token.type=='table_open':
                rows=[t for t in parsed if t.type=='tr_open' and t.map and token.map[0]<=t.map[0]<token.map[1]]
                if len(rows)<2:
                    output.append(('unresolved_table',[(a,b,'body')]));continue
                starts=[offsets[t.map[0]] for t in rows[1:]]
                if self.count(prefix+text[a:starts[0]])>=self.maximum:
                    output.append(('unresolved_table_header_budget',[(a,b,'body')]));continue
                output.extend(('table_rows',r) for r in self._pack(text,list(zip(starts,starts[1:]+[b])),prefix,[(a,starts[0],'body')]))
            elif token.type in {'bullet_list_open','ordered_list_open'}:
                items=[t for t in parsed if t.type=='list_item_open' and t.level==1 and t.map and token.map[0]<=t.map[0]<token.map[1]]
                starts=[a]+[offsets[t.map[0]] for t in items[1:]]
                output.extend(('list_items',r) for r in self._pack(text,list(zip(starts,starts[1:]+[b])),prefix))
            elif any(t.type in {'fence','code_block','table_open'} and t.map and a<=offsets[t.map[0]]<b for t in parsed):
                output.append(('unresolved_protected_container',[(a,b,'body')]))
            else:
                output.extend(self._plain(text,a,b,prefix))
        flush()
        return output

    def _hard_plan(self, text, prefix, preferred):
        """Exact source continuations: structures -> statements -> lines -> Tokens.

        Fences and rows may cross a continuation boundary. No synthetic closing
        fence, indentation, pass statement or class/function shell is inserted.
        """
        if self.count(prefix)>=self.maximum:raise ValueError('unresolved_heading_budget')
        points={0,len(text)}
        for _,ranges in preferred:
            points.update(x for a,b,role in ranges if role=='body' for x in (a,b))
        statement_points=set()
        offsets=[0]
        for line in text.splitlines(True):offsets.append(offsets[-1]+len(line))
        for token in MarkdownIt().parse(text):
            if token.map:
                statement_points.update((offsets[token.map[0]],offsets[token.map[1]]))
            if token.type=='fence' and token.info.split() and token.info.split()[0] in {'py','python'}:
                try:
                    tree=ast.parse(token.content)
                    for node in ast.walk(tree):
                        if isinstance(node,ast.stmt):statement_points.add(offsets[token.map[0]+node.lineno])
                except (SyntaxError,ValueError,RecursionError,IndexError):pass
        statement_points.update(m.end() for m in re.finditer(r'\n\s*\n',text))
        units=[]
        def split(a,b,level):
            if self.count(prefix+text[a:b])<=self.maximum:
                units.append((a,b));return
            cuts=sorted({a,b,*[p for p in (statement_points if level==0 else offsets) if a<p<b]})
            if level<2 and len(cuts)>2:
                for x,y in zip(cuts,cuts[1:]):split(x,y,level+1)
                return
            if level<1:
                split(a,b,1);return
            proposal=self._plain(text,a,b,prefix)
            if all(not method.startswith('unresolved') for method,_ in proposal):
                units.extend((x,y) for _,rs in proposal for x,y,_ in rs);return
            # Character offsets, never decoded Token IDs: even a split inside
            # a Unicode word/long literal must preserve the original bytes.
            cursor=a
            while cursor<b:
                lo,hi=cursor+1,b;best=cursor
                while lo<=hi:
                    end=(lo+hi)//2
                    if self.count(prefix+text[cursor:end])<=self.target:best=end;lo=end+1
                    else:hi=end-1
                if best==cursor:
                    if self.count(prefix+text[cursor:cursor+1])>self.maximum:
                        raise ValueError('Input prefix/heading leaves no room for one source character')
                    best=cursor+1
                units.append((cursor,best));cursor=best
        ordered=sorted(points)
        for a,b in zip(ordered,ordered[1:]):split(a,b,0)
        groups=self._pack(text,units,prefix)
        # Attach whitespace-only tails without dropping even one source space.
        for i in range(len(groups)-1,-1,-1):
            if not self._text(text,groups[i]).strip():
                if i:groups[i-1].extend(groups.pop(i))
                elif len(groups)>1:
                    blank=groups.pop(0);groups[0]=blank+groups[0]
        if any(self.count(prefix+self._text(text,g))>self.maximum for g in groups):
            # No accepted over-budget result: report a hard failure to exporter.
            raise ValueError('Lossless continuation packing exceeded the hard Token limit')
        return [('source_continuation',g) for g in groups]

    def refine(self, section, chunks):
        output=[]
        source_owned=bytearray(len(section.content))
        def parent_parts(c):
            return c.content_parts or [ChunkPart(start=0,end=len(c.content),section_start=c.section_start,section_end=c.section_end)]
        def mark_parent(c):
            for p in parent_parts(c):
                if p.role=='body':source_owned[p.section_start:p.section_end]=b'\1'*(p.section_end-p.section_start)
        next_index=max((c.chunk_index for c in chunks),default=-1)+1
        for original in chunks:
            if self.input_prefix:
                original=original.model_copy(update={'embedding_text':self.input_prefix+original.embedding_text})
            before=self.count(original.embedding_text)
            if before<=self.maximum:
                output.append(original);mark_parent(original);continue
            if not original.embedding_text.endswith(original.content):
                raise ValueError('embedding_text must end with the exact Chunk content')
            prefix=original.embedding_text[:-len(original.content)]
            plan=self._plan(original.content,prefix) if self.count(prefix)<self.maximum else [('unresolved_heading_budget',[(0,len(original.content),'body')])]
            if any(self.count(prefix+self._text(original.content,r))>self.maximum for _,r in plan):
                try:plan=self._hard_plan(original.content,prefix,plan)
                except ValueError as exc:
                    self.findings.append(dict(chunk_id=original.chunk_id,parent_chunk_id=original.chunk_id,source_path=original.source_path,
                        section_id=original.section_id,tokens=before,reason=str(exc)))
                    output.append(original);mark_parent(original);continue
            coverage=bytearray(len(original.content))
            for _,ranges in plan:
                for a,b,_ in ranges:coverage[a:b]=b'\1'*(b-a)
            if not all(coverage):raise ValueError('Token refinement would lose original text')
            claimed=bytearray(len(original.content));last_end=0
            for _,ranges in plan:
                for a,b,role in ranges:
                    if role!='body':continue
                    fresh=[j for j in range(a,b) if not claimed[j]]
                    if fresh and fresh[0]<last_end:raise ValueError('Token fragments reorder original body text')
                    if fresh:last_end=fresh[-1]+1
                    claimed[a:b]=b'\1'*(b-a)
            if len(plan)==1 and self._text(original.content,plan[0][1])==original.content:
                output.append(original)
                mark_parent(original)
                self.findings.append(dict(chunk_id=original.chunk_id,parent_chunk_id=original.chunk_id,source_path=original.source_path,
                    section_id=original.section_id,tokens=before,reason=plan[0][0] if plan[0][0].startswith('unresolved') else 'indivisible_structure'))
                continue
            owned=bytearray(len(original.content));children=[];lineage=[]
            source_parts=parent_parts(original)
            for i,(method,ranges) in enumerate(plan):
                pieces,parts,spans=[],[],[];cursor=0
                for a,b,role in ranges:
                    # Mark repeated prose/list/function units as context, never
                    # as a second ownership claim on the same source characters.
                    cuts=[a]+[j for j in range(a+1,b) if owned[j]!=owned[j-1]]+[b]
                    for x,y in zip(cuts,cuts[1:]):
                        actual_role='context' if role=='body' and owned[x] else role
                        for p in source_parts:
                            lo,hi=max(x,p.start),min(y,p.end)
                            if lo>=hi:continue
                            sa,sb=p.section_start+lo-p.start,p.section_start+hi-p.start
                            if section.content[sa:sb]!=original.content[lo:hi]:raise ValueError('Invalid parent Chunk source mapping')
                            mapped_role=p.role if p.role!='body' and actual_role=='body' else actual_role
                            boundaries=([sa]+[j for j in range(sa+1,sb) if source_owned[j]!=source_owned[j-1]]+[sb]) if mapped_role=='body' else [sa,sb]
                            for sx,sy in zip(boundaries,boundaries[1:]):
                                final_role='context' if mapped_role=='body' and source_owned[sx] else mapped_role
                                piece=section.content[sx:sy]
                                pieces.append(piece);parts.append(ChunkPart(start=cursor,end=cursor+len(piece),section_start=sx,section_end=sy,role=final_role))
                                spans.extend(s.model_copy(update={'start':s.start+cursor,'end':s.end+cursor}) for s in slice_spans(section.source_spans,sx,sy))
                                cursor+=len(piece)
                            if mapped_role=='body':source_owned[sa:sb]=b'\1'*(sb-sa)
                    if role=='body':owned[a:b]=b'\1'*(b-a)
                content=''.join(pieces)
                if content!=self._text(original.content,ranges):raise ValueError('Rendered token fragment differs from exact parent slices')
                if not content.strip():raise ValueError('Token refinement produced an empty Chunk')
                embedding=prefix+content
                types=[]
                kinds={'fence':'code','code_block':'code','table_open':'table','bullet_list_open':'list','ordered_list_open':'list','heading_open':'heading','paragraph_open':'paragraph'}
                for t in MarkdownIt().enable('table').parse(content):
                    if t.type in kinds and kinds[t.type] not in types:types.append(kinds[t.type])
                    if t.children and any(c.type=='image' for c in t.children) and 'image' not in types:types.append('image')
                if method=='source_continuation':types=list(dict.fromkeys([*original.block_types,*types]))
                identity=stable_hash([TOKEN_CHUNKER_VERSION,original.chunk_id,self.fingerprint,self.config,method,[p.model_dump() for p in parts],content])
                child=original.model_copy(update=dict(chunk_id=identity,chunk_index=original.chunk_index if i==0 else next_index,
                    content=content,embedding_text=embedding,block_types=types or original.block_types,content_parts=parts,
                    source_spans=spans,section_start=min(p.section_start for p in parts),section_end=max(p.section_end for p in parts),
                    split_method='token_'+method,oversized=len(content)>self.chunk_size))
                if i:next_index+=1
                children.append(child)
                lineage.append(dict(chunk_id=identity,parent_ranges=[dict(start=a,end=b,role=role) for a,b,role in ranges]))
                n=self.count(embedding)
                if n>self.maximum:
                    self.findings.append(dict(chunk_id=identity,parent_chunk_id=original.chunk_id,source_path=original.source_path,
                        section_id=original.section_id,tokens=n,reason=method if method.startswith('unresolved') else 'indivisible_'+method))
            output.extend(children)
            self.changes.append(dict(parent_chunk_id=original.chunk_id,before_tokens=before,child_chunk_ids=[c.chunk_id for c in children]))
            restored=['']*len(original.content)
            for child,(_,ranges) in zip(children,plan):
                cursor=0
                for a,b,_ in ranges:
                    restored[a:b]=child.content[cursor:cursor+b-a];cursor+=b-a
            if ''.join(restored)!=original.content:raise ValueError('Continuation reassembly differs from parent content')
            self.reconstructions.append(dict(parent_chunk_id=original.chunk_id,original_sha256=hashlib.sha256(original.content.encode('utf-8')).hexdigest(),
                reconstructed_sha256=hashlib.sha256(''.join(restored).encode('utf-8')).hexdigest(),children=lineage,continuation=any(m=='source_continuation' for m,_ in plan)))
        return output
