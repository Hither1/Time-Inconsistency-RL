penalty = .1

print("AT state 25")
res = sum([-1 * penalty * 1/(1 + i) for i in range(1, 3 + 1)]) + 1/(1 + 3) * 6
print(res)

res = sum([-1 * penalty * 1/(1 + i) for i in range(1, 5 + 1)]) + 1/(1 + 5) * 10
print(res)