import socket
import struct
import pickle

BUFFER_SIZE = 4096

class Communicate():
	def __init__(self, TCP_IP, TCP_PORT):
		self.TCP_IP = TCP_IP
		self.TCP_PORT = TCP_PORT
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.clients = {}
		self.count = 1
		self.payload_size = struct.calcsize("I")

	def receive(self, addr):
		try: 
			if self.sock.connect_ex(addr):
				conn = self.clients[addr]
		except:
			conn, addr = self.sock.accept()
			self.clients[addr] = conn

		print("Got connection from",addr)
		print("Receiving... {}".format(self.count))

		data = conn.recv(BUFFER_SIZE)

		self.count+=1

		return data, addr

	def send(self, addr, data):
		self.clients[addr].sendall(data)
		self.clients[addr].close()
		del self.clients[addr]
		print("Send data to '{}'".format(addr))

	def listen(self):
		self.sock.bind((self.TCP_IP, self.TCP_PORT))
		self.sock.listen(3)
		print("Listening...")

