import time
start_time = time.time()
# your script
for i in range(100000000):
	x = (i*i)
elapsed_time = time.time() - start_time
print_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print(print_time)