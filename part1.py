import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import timeit

#x = np.arange(1,151)


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


if __name__ == '__main__':

	accuracy = 10 ** -3
	number = np.zeros(12)
	temp=2000
	upper=4000
	lower=0
	for n in range(12):
	    step = int(temp)
	    temp=temp-lower
	    while(True):
	        temp=int(temp/2)
	        print(step)
	        pre=step
	        value_now = Binomial("P", 100, (n + 1) / 12, 100, 0.2, 0.05, 0.04, step, "A")

	        value_next = Binomial("P", 100, (n + 1) / 12, 100, 0.2, 0.05, 0.04, step+1, "A")
	        if abs(value_next - value_now) > accuracy:
	            step=step+temp
	        else:
	            step=step-temp
	        
	        if(step==pre):
	            break
	    
	    number[n] = step
	    lower=step
	    temp=(upper-lower)/2+lower
	    #print("month ")
	print(number)

	plt.scatter(x1 + 1, y1)
	plt.xlabel("Time to Maturity (Months)")
	plt.ylabel("Number of Time Steps")
