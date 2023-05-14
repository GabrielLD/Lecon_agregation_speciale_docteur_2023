def justify_paragraph(string, width=40):
    out = ['']
    in_equation = False
    for word in string.split(' '):
        if word.startswith('$'):
            in_equation = True
        if not in_equation:
            if len(out[-1]) + len(word) + 1> width:
                out.append('')
        out[-1] += ' ' + word
        if word.endswith('$'):
            in_equation = False
    out = [elm for elm in out if elm.strip()] # Remove blanck lines
    return '\n'.join(out)

def justify(string, width=40):
    paragraphs = ['']
    for lines in string.split('\n'):
        if lines.strip()=='':
            paragraphs.append('')
        else:
            paragraphs[-1] += lines.strip() + ' '

    out = [justify_paragraph(paragraph, width=width) for paragraph in paragraphs]
    return '\n\n'.join(out)
    
