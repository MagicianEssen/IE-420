import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import timeit

x = np.arange(1,151)


def Binomial(Option, K, T, S0, sigma, r, q, N, Exercise):
    if (Option=="C")and (Exercise=="E"):
        delta = T/N
        u = np.e**(sigma*np.sqrt(delta))
        d = np.e**(-sigma*np.sqrt(delta))
        p = (np.e**((r-q)*delta)-d)/(u-d)
        blue = np.zeros(N+1)
        for i in range(N+1):
            blue[i] = S0 * (u**(N-i)) * (d**(i))
        blue = (blue - K).clip(min=0)

        for i in range(N):
            temp = np.zeros(N - i)
            for x in range(N - i):
                temp[x] = np.e**((-r) * delta) * (1 * (p) * blue[x] + (1-p) * blue[x+1])
            blue = temp
        return blue
    if (Exercise=="A") and (Option=="P"):
        delta = T / N
        u = np.e ** (sigma * np.sqrt(delta))
        d = np.e ** (-sigma * np.sqrt(delta))
        p = (np.e ** ((r - q) * delta) - d) / (u - d)
        blue = []



        for i in range(N + 1):
            temp = []
            for j in range(i + 1):
                temp.append(S0 * (u ** (i - j)) * (d ** (j)))
            blue.append(temp)
        black = (K - np.asarray(blue[-1])).clip(min=0)
        for i in range(N):
            for x in range(N - i):
                black[x] = max(np.e**((-r)*delta)*((p)*black[x]+(1-p)*black[x+1]),
                               K-blue[N-i-1][x])
        return black[0]
    if (Exercise=="A") and (Option=="C"):
        delta = T / N
        u = np.e ** (sigma * np.sqrt(delta))
        d = np.e ** (-sigma * np.sqrt(delta))
        p = (np.e ** ((r - q) * delta) - d) / (u - d)
        blue = []
        for i in range(N + 1):
            temp = []
            for j in range(i + 1):
                temp.append(S0 * (u ** (i - j)) * (d ** (j)))
            blue.append(temp)
        black = (np.asarray(blue[-1]) - K).clip(min=0)
        for i in range(N):
            for x in range(N - i):
                black[x] = max(np.e ** ((-r) * delta) * (1 * (p) * black[x] + (1 - p) * black[x + 1]),
                               blue[N - i - 1][x] - K)
        return black[0]


number = np.zeros(12)

number[0]=1104
number[1]=1527
number[2]=1840
number[3]=2086
number[4]=2292
number[5]=2476
number[6]=2637
number[7]=2780
number[8]=2908
number[9]=3027
number[10]=3131
number[11]=3234



guesstar = np.zeros(12)
upper=100
lower=50
temp=25
for i in range(12):

    if(i!=0):
        upper = guesstar[i-1]

    temp=(upper-lower)/2
    guess=lower+temp
    while (True):

        temp=temp/2
        pre=guess
        print(guess)
        option_price = Binomial("P", 100, (i+1)/12, guess, 0.2, 0.05, 0.04,int( number[i]), "A")
        if((option_price + guess -100) > 0.005):
            guess =guess-temp
        else:
            guess =guess+temp
        
        if(int(guess*100)==int(pre*100)):
            break

    guesstar[i]=int(guess*100)/100
    upper=guess
    temp=(upper-lower)/2
            
    print("month ", i, " :", guesstar[i])
print(guesstar)


plt.scatter(x1+1,guesstar)
plt.xlabel("Time to Maturity (Months)")
plt.ylabel("Critical Stock Price (P) ")


