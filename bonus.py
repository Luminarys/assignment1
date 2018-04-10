import autodiff as ad
import numpy as np
import csv

EPOCHS = 3
LEARN_RATE = 0.1

def train(weights, bias, data):
    b = ad.Variable(name = "b")
    w = ad.Variable(name = "w")
    x = ad.Variable(name = "x")

    y = 1 / (1 + ad.exp_op(-1 * (b + (w @ x))))
    w_grad, b_grad = ad.gradients(y, [w, b])
    executor = ad.Executor([y, w_grad, b_grad])

    dw = np.zeros(np.shape(data[0][1:]))
    db = 0.
    err = 0.
    data = data[0:15]
    for d in data:
        print(d)
        vals = {b: bias, w: weights, x: d[1:]}
        y_val, grad_vals, bias_grad_val = executor.run(feed_dict = vals)
        print(y_val, grad_vals, bias_grad_val)
        y_err = d[0] - y_val
        dw += y_err * grad_vals * d[1:]
        db += y_err * bias_grad_val * 1.
        err += np.abs(y_err)

    print(dw)
    print(db)
    return (dw/len(data), db/len(data), err/len(data))

def main():
    data = np.genfromtxt('./binary.csv', delimiter=',')[1:]
    weights = np.array([0., 0., 0.])
    bias = 0.

    for i in range(0, EPOCHS):
        dw, db, err = train(weights, bias, data)
        weights += LEARN_RATE * dw
        bias += LEARN_RATE * db
        print("EPOCH {}, err: {}, weights {}, bias {}".format(i, err, weights, bias))

if __name__ == "__main__":
    main()
