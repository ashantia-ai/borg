from SimpleXMLRPCServer import SimpleXMLRPCServer
import xmlrpclib

import multiprocessing
import time
import random
import os
import netifaces as ni

global process
process = None

log_path = os.environ['HOME'] +"/cacla_log.txt"
def dummy_process(n):
    print "Process is running"
    time.sleep(n)
    
    
def run_cacla_navigation(params):
    import cacla_nav
    if os.environ['HOME'] == "/home/work-pc":
        base_path = "/home/work-pc/nav-data"
    else:
        base_path = "/media/Datastation/nav-data-late14/BACKUPSTUFF"
    reference_frame = 'odom'
    
    print repr(params)
    
    
    nav = cacla_nav.CACLA_nav(base_path, reference_frame, **params)
    nav.spin()
    nav.reset()
    


def is_finished():
    if process.is_alive():
        return False
    else:
        print "Process has finished is job"
        return True
     
def start(params):
    global process
    
    f = open(log_path, 'a')
    
    try:
        process = multiprocessing.Process(target=run_cacla_navigation, args=(params,))
        process.start()
    except Exception as e:
        f.write(e)
        raise
    return True

net_card = None
for i in range(4):
    try:
        net_card = ni.ifaddresses('eth'+str(i))
    except:
        pass
ip = net_card[2][0]['addr']
print "Server Ethernet Address is ", ip 
server = SimpleXMLRPCServer((ip, 8000), logRequests=False)
print "Listening on port 8000..."
server.register_function(is_finished, 'is_finished')
server.register_function(start, 'start')

server.serve_forever()
