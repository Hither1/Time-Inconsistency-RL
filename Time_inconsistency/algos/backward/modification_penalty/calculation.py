#
penalty = .1
result = sum([-1 * penalty / (1 + i) for i in range(1, 8+1)]) + 19 * 1/(1 + 1*8)
print(result)
result = sum([-1 * penalty / (1 + i) for i in range(1, 6+1)]) + 19 * 1/(1 + 1*6)
print(result)
result = sum([-1 * penalty / (1 + i) for i in range(1, 4+1)]) + 10 * 1/(1 + 1*4)
print(result)

# only after a certain length of episode
penalty = .1
result = sum([-1 * penalty / (1 + i) for i in range(4, 8+1)]) + 19 * 1/(1 + 1*8)
print(result)
result = sum([-1 * penalty / (1 + i) for i in range(4, 6+1)]) + 19 * 1/(1 + 1*6)
print(result)
result = sum([-1 * penalty / (1 + i) for i in range(4, 4+1)]) + 10 * 1/(1 + 1*4)
print(result)

# variable
penalty = .05
result = sum([-1 * i * penalty / (1 + i) for i in range(1, 8+1)]) + 19 * 1/(1 + 1*8)
print(result)
result = sum([-1 * i * penalty / (1 + i) for i in range(1, 6+1)]) + 19 * 1/(1 + 1*6)
print(result)
result = sum([-1 * i * penalty / (1 + i) for i in range(1, 4+1)]) + 10 * 1/(1 + 1*4)
print(result)