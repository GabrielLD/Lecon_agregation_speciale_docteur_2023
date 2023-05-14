import subprocess

from .index import tous_les_programmes

lines = []
for programme_name in tous_les_programmes:
    _temp = __import__(programme_name, globals(), locals(), ['fig'], 0)
    _temp.fig.savefig('_pdf/{}.pdf'.format(programme_name))
    lines.append('* {titre} (`{name}.py <{name}.py>`_)'.format(titre=_temp.titre.strip(), name=programme_name))

print('\n'.join(lines))

