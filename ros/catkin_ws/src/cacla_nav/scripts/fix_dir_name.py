import os

for dirname, dirnames, filenames in os.walk('.'):
    # print path to all subdirectories first.
    for subdirname in dirnames:
        if subdirname == 'nfq' or 'goal' in subdirname or 'net' in subdirname or 'multi' in subdirname:
            continue
        first_part = float(subdirname.split("_",1)[0])
        second_part = float(subdirname.split("_",1)[1])
        newname = "{:.1f}".format(first_part) + '_' + "{:.1f}".format(second_part)
        os.rename('./' + os.path.join(dirname, subdirname), './' + os.path.join(dirname, newname))  
        print './' + os.path.join(dirname, subdirname)
        print './' + os.path.join(dirname, newname)
