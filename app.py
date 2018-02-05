# -*- coding: utf-8 -*- 
#!flask/bin/python
from flask import Flask, jsonify
from flask import request
import socket
from flask_cors import CORS
import re
import config as cfg
import string

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

filename = 'answers.txt'
answer_dict = {}

TCP_IP = "121.134.144.52"#"10.214.35.36"
TCP_PORT = 5005

@app.route('/api/v1.1/chat',methods=['GET'])
def get_sentence():
    """
    API: get user's query sentence
    ex) 121.134.144.52:5000/api/v1.1/chat?sentence=QUERY
    """
    try:
        qa = request.args.get('sentence')
        address = (TCP_IP,TCP_PORT)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(address)
        send_question(s, qa)
        answer = recv_answer(s)
    except:
        answer = "잘못된 URL일 가능성이 있습니다."

    try:
        answer = match_answer(answer)
            
    except:
        answer = "이해할 수 없는 단어가 있습니다."
	
    if answer[:1] != '{':
        answer = "{\"answer\":\""+answer.rstrip('\n')+"\"}"
	
    s.close()
    print(answer)
    return answer

def get_json(answer):
    json_dict = []
    splited = answer.split(' ')
    for s in splited:
        s = s.split(':')[0]
        json_dict.append(s)

    return json_dict

def type_check(answer):
    """
    check the answer type which is QA or Search
    type 1 is QA sentence
    type 2 is Search sentence
    """
    length = len(answer.split(' '))
    
    if length == 1:
        return 1 #QA
    else:
        return 2 #Search

def match_answer(code):
    """
    search ANSWER CODE from answer_dict
    """
    code = re.split(':| ',code)
    return answer_dict[code[0]+code[2]]

def make_answer_dict(filename):
    """
    make answer dict with answer.txt file
    """
    with open(filename,'r',encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            (key,val) = line.split('\t')
            answer_dict[key] = val;

def send_question(sock, q):
    """
    send user's query to analyze server by tcp socket
    """
    sock.sendall(q.encode())
    sock.shutdown(socket.SHUT_WR)

def recv_answer(sock):
    """
    receive analyized answer code from server
    """
    a = sock.recv(cfg.BUFFER_SIZE)
    a = a.decode()
    return a

def sub_numbers_to_char(s):
    """
    substitue numbers in sentence to random character
    ex) 2L 백산수 검색해줘 -> X L 백산수 검색해줘
    """
    numbers = [s for s in str.split() if s.isdigit()]

    for number in numbers:
        letter = random.choice(string.letters)
        s.replace(letter,number)

    return s
    

if __name__ == '__main__':
    make_answer_dict(filename)
    app.run(host="0.0.0.0")#app.run(host="10.214.35.36")
