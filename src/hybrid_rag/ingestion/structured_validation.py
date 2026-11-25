"""Independent checks of rendered fragments, in addition to exact source parts."""
import ast
from markdown_it import MarkdownIt


def table_values(text):
    tokens = MarkdownIt().enable('table').parse(text)
    header, rows, row = [], [], []
    in_head = False
    for token in tokens:
        if token.type == 'thead_open': in_head = True
        elif token.type == 'thead_close': in_head = False
        elif token.type == 'tr_open': row = []
        elif token.type == 'inline': row.append(token.content)
        elif token.type == 'tr_close':
            if in_head: header = row
            else: rows.append(row)
    return header, rows


def validate_structure(section, chunks, token, start, end):
    related = [c for c in chunks if c.content_parts and any(p.section_start < end and p.section_end > start for p in c.content_parts)]
    source = section.content[start:end]
    errors = []
    if token.type == 'fence':
        fragments = []
        for chunk in related:
            fences = [t for t in MarkdownIt().parse(chunk.content) if t.type == 'fence']
            if len(fences) != 1 or fences[0].info != token.info:
                errors.append('split_code_fence_structure'); continue
            try: ast.parse(fences[0].content)
            except (SyntaxError, ValueError, RecursionError): errors.append('split_python_invalid')
            fragments.append(fences[0].content)
        if ''.join(fragments) != token.content:
            errors.append('split_code_body_changed')
    elif token.type == 'table_open':
        header, expected = table_values(source)
        actual = []
        for chunk in related:
            h, rows = table_values(chunk.content)
            if h != header: errors.append('split_table_header_changed')
            actual.extend(rows)
        if actual != expected: errors.append('split_table_rows_changed')
    return errors


def validate_token_structure(section, chunks, token, start, end):
    """Validate source ownership and rendered structures, allowing named overlap."""
    from hybrid_rag.schemas.document import ChunkPart
    errors=[]
    offsets=[0]
    for line in section.content.splitlines(True):offsets.append(offsets[-1]+len(line))
    a,b=(offsets[token.map[0]+1],offsets[token.map[1]-1]) if token.type=='fence' else (start,end)
    ownership=bytearray(b-a)
    for c in chunks:
        parts=c.content_parts or [ChunkPart(start=0,end=len(c.content),section_start=c.section_start,section_end=c.section_end)]
        intersections=[(p,max(a,p.section_start),min(b,p.section_end)) for p in parts if p.section_start<b and p.section_end>a]
        if not intersections:continue
        expected=''.join(section.content[x:y] for p,x,y in intersections)
        for p,x,y in intersections:
            if p.role=='body':
                for i in range(x-a,y-a):ownership[i]=min(2,ownership[i]+1)
        if c.split_method=='token_source_continuation':
            # Exact slices and single source ownership are checked above and
            # by Validator. A declared continuation can cross a fence/row.
            continue
        parsed=MarkdownIt().enable('table').parse(c.content)
        if token.type=='fence':
            opening=section.content.splitlines()[token.map[0]]
            margin=opening[:opening.find(token.markup)]
            if expected==section.content[a:b]:
                expected=token.content  # original parser removes list/quote container margins
            elif not margin.strip():
                expected=''.join(line[min(len(margin),len(line)-len(line.lstrip(' '))):] for line in expected.splitlines(True))
            candidates=[t for t in parsed if t.type=='fence' and t.info==token.info and t.content==expected]
            if not candidates:errors.append('token_code_fence_or_body_changed')
            for t in candidates:
                lines=c.content.splitlines()
                import re
                if not re.fullmatch(re.escape(t.markup[0])+'{'+str(len(t.markup))+',}',lines[t.map[1]-1].strip()):errors.append('token_unclosed_fence')
                if c.split_method=='token_python_ast':
                    try:ast.parse(t.content)
                    except (SyntaxError,ValueError,RecursionError):errors.append('token_python_invalid')
        else:
            wanted=table_values(expected)
            lines=c.content.splitlines(True)
            tables=[table_values(''.join(lines[t.map[0]:t.map[1]])) for t in parsed if t.type=='table_open' and t.map]
            if wanted not in tables:errors.append('token_table_rendering_changed')
    if any(n!=1 for n in ownership):errors.append('token_structure_body_lost_or_duplicated')
    return errors
