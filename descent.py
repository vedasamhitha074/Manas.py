from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def descent(x, y, lr=0.1, epochs=1000):
   
    n=len(x)
    m=0.0
    c=0.0

    
    for i in range(epochs):
        #z= predicted value of y
        z=(m*x)+c
        
        dm=(-2/n)*np.sum(x*(y-z))
        dc=(-2/n)*np.sum(y-z)
        m-=lr*dm
        c-=lr*dc
    return m,c,
def main():
    file_path ="weatherHistory.csv"
    df=pd.read_csv(file_path)

    data=df[["Humidity", "Temperature (C)"]].dropna()
    x=data["Humidity"].values
    y=data["Temperature (C)"].values

    m,c,=descent(x, y,lr=0.1,epochs=3000)

    print("Gradient descent results:")
    print(f"Slope (m)     : {m:.4f}")
    print(f"Intercept (c) : {c:.4f}")
    print(f"Eqn           : Temp = ({m:.2f} * Humidity) + {c:.2f}")

    plt.scatter(x, y, alpha=0.3, label="Data points")
    # line predictions
    z=(m*x)+c
    plt.plot(x, z, color="red", linewidth=2, label="Fit line")
    plt.xlabel("Humidity")
    plt.ylabel("Temperature (C)")
    plt.title("Humidity vs Temperature")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
  
