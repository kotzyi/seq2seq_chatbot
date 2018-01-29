# -*- coding: utf-8 -*- 
#!flask/bin/python
from flask import Flask, jsonify
from flask import request
import socket
from flask_cors import CORS
import re

TCP_IP = "121.134.144.52"
TCP_PORT = 5005
BUFFER_SIZE  = 4096

#Alpha = {'WGT':'1.8KG', 'VOL':'2L', 'TMP':'100c', 'WRM':'가능', 'TIM':'가능', 'MTR':'스테인레스', 'SIZ':'22x18x40'}
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

filename = 'answers.txt'
answer_dict = {}

#@crossdomain(origin='*', headers='Content-Type, X-User, X-Token')
@app.route('/api/v1.0/qa',methods=['GET'])
def get_qa():
    sentence = request.args.get('sentence')
    address = (TCP_IP,TCP_PORT)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(address)

    send_question(s, sentence)
    answer = recv_answer(s)
    try:
        answer = match_answer(answer)
    except:
        answer = "이해할 수 없는 단어가 있습니다."
    answer = "{\"answer\":\""+answer+"\"}"
    s.close()
    print(answer)
    return answer

@app.route('/api/v1.0/search',methods=['GET'])
def get_search():
    search = request.args.get('search')
    address = (TCP_IP,TCP_PORT)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(address)

    send_question(s, search)
    answer = recv_answer(s)
    try:
        answer = match_answer(answer)
    except:
        answer = "이해할 수 없는 단어가 있습니다."
    
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
    length = len(answer.split(' '))
    
    if length == 1:
        return 1 #QA
    else:
        return 2 #Search

#@app.errorhandler(404)
#def not_found(error):
#    return make_response(jsonify({'error': 'Not Found'}), 404)
def match_answer(code):
    code = re.split(':| ',code)
    return answer_dict[code[0]+code[2]]

def make_answer_dict(filename):
    with open(filename,'r',encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            (key,val) = line.split('\t')
            answer_dict[key] = val;

def send_question(sock, q):
    sock.sendall(q.encode())
    sock.shutdown(socket.SHUT_WR)

def recv_answer(sock):
    a = sock.recv(BUFFER_SIZE)
    a = a.decode()
    return a

def recon_sentence(sentence):
    words = []
    sentence = sentence.split(' ')

    for s in sentence:
        w = s.split(":")
        print (w)
        try:
            if w[1] != 'Punctuation':
                if w[1] == 'Alpha':
                    w[0] = Alpha[w[0]]

                words.append(w[0])
        except:
            pass

    return ' '.join(words)

if __name__ == '__main__':
    make_answer_dict(filename)
    app.run(host="0.0.0.0")
