from subprocess import Popen, PIPE
from io import BufferedReader, BufferedWriter

#reader = BufferedReader()
#writer = BufferedWriter()
#proc = Popen('cmd.exe', stdin=reader, stdout=writer)
proc = Popen('nslookup.exe', stdin=PIPE, stdout=PIPE, shell=True)
writer = proc.stdin
reader = proc.stdout
print(reader.read(1))
