import random
import time
import json
import pandas as pd
from datetime import datetime 


NOW = datetime.now()


def gen_phone_number():
    list = [0,1,2,3,4,5,6,7,8,9]
    number = "09"
    for i in range(0,9):
        addition = random.choice(list)
        number += str(addition)
    return number

    
def gen_with_dups(number_of_phones, duplicate_probability, max_duplicates_per_number):
    phone_numbers = []
    unique_numbers = {} 
    
    for _ in range(number_of_phones):
        target_duplicates = int(random.random() * max_duplicates_per_number) 
        unique_numbers[gen_phone_number()] = target_duplicates
        
    for new_number, target_count in unique_numbers.items():
        actual_count = 0
        while actual_count < target_count:
            phone_numbers.append(new_number)
            actual_count += 1

            if actual_count < target_count and random.random() > duplicate_probability:
                break
    return phone_numbers


phone_numbers = gen_with_dups(1000,0.6,30)
phone_numbers = random.sample(phone_numbers,len(phone_numbers))


def gen_price():
    base = range(0,1)
    cost = ''
    for i in base:
        random_num = random.randint(30000,20000000)
        cost += str(random_num)
    return cost
    
def str_time_prop(start, end, time_format, prop):

    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(time.strptime(end, time_format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(time_format, time.localtime(ptime))


def random_date(start, end, prop):
    return str_time_prop(start, end, '%Y-%m-%d %I:%M %p', prop)

index_list = [index for index, item in enumerate(phone_numbers)]

def json_creator(): 
    lis = []
    for i , a in zip(phone_numbers,index_list):
        sample = {
                    "pk": a ,
                    "fields": {
                        "phone": i ,
                        "price": gen_price(),
                        "date": random_date("2022-01-01 1:30 PM", "2024-03-25 4:50 AM", random.random()),
                        "order_condition":'4',
                    }
                }
        lis.append(sample)  
    return lis

sample_file = json_creator()
json_file = json.dumps(sample_file , indent=2)

f = open("sample.json", "w")
f.write(json_file)
f.close()
 
with open('sample.json', 'r') as j:
    data = json.loads(j.read())
df = pd.json_normalize(data)
df.to_csv('sample.csv', index=False)

# print(json_creator())
