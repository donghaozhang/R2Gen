import icd10

# code = icd10.find("J20.0")
# print(code.description)         # Acute bronchitis due to Mycoplasma pneumoniae
# if code.billable:
#     print(code, "is billable")  # J20.0 is billable

# print(code.chapter)             # X
# print(code.block)               # J00-J99
# print(code.block_description)   # Diseases of the respiratory system

# if icd10.exists("J20.0"):
#     print("Exists")


# eye_code = icd10.find('H01.0')
eye_code = icd10.find('H25')
print(eye_code.chapter)
print(eye_code.block)
print(eye_code.block_description)
print('eye_code', eye_code)
# eye_code.get_description("XII")

# for i in range(1, 6):
# 	# print(i)
# 	for j in range(10):
# 		# print(j)
# 		curstr = 'H' + str(i) + str(j) + '.0'
# 		print(curstr)
# 		eye_code = icd10.find('H'+curstr)
# 		print(eye_code.block_description)
