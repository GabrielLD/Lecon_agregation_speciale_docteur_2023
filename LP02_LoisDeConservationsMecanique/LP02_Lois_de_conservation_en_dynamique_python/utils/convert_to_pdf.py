import subprocess

from .index import tous_les_programmes

for programme_name in tous_les_programmes:
    print(programme_name, end='')
    _temp = __import__(programme_name, globals(), locals(), ['fig'], 0)
    _temp.fig.savefig('_pdf/{}.pdf'.format(programme_name))
    print(': OK')
#    print('* {}'.format(_temp.titre))

file_list = ['_pdf/{}.pdf'.format(programme_name) for programme_name in tous_les_programmes]

try:
    subprocess.call(['pdfjoin', '--outfile', 'doc/programmes_lecons.pdf']+file_list)
except Exception as e:
    msg = 'Not able to merge pdf files in the single file. Individual files are in the _pdf folder'
    print('#'*len(msg))
    print(msg)
    print('#'*len(msg))
    raise

