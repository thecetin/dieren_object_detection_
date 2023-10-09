import json
from threading import Thread
import time

class readJsonFile:
    def __init__(self,r_time):
        self.r_time = r_time
        self.__enter__()

    #def __enter__(self)
    def __enter__(self):
        self.f = open('/home/user/env/example.json','r')
        self.data = json.load(self.f)
    
    def check_seconds(self):
        recievedd_sec = int(self.r_time) #51604 #14:10:11
        #print(self.r_time)
        for i in range(0,len(self.data)):
            d = dict(self.data[i])
            #print("dictinary ",d)
            #print(d['created_at'])
            t = str(d['created_at'])
            t = t[11:-5]
            sec = int(t[6:])
            min = 60* int(t[3:-3])
            hour = 60*60*int(t[:-6])
            self.seconds = sec+min+hour
            print("Okunan = ",self.seconds)
            if(self.seconds+5 >= recievedd_sec and self.seconds-5 <=recievedd_sec):
                print("time: ",t)
                return self.seconds
            
    
    def __exit__(self):
        self.f.close()

read = readJsonFile('51609')
print("Result",read.check_seconds())