from question3 import y_te, y_pred_gd, y_pred_ne, y_pred_sk

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(y_te, y_pred_gd)
plt.scatter(y_te, y_pred_ne)
plt.scatter(y_te, y_pred_sk)

plt.plot([min(y_te), max(y_te)],
         [min(y_te), max(y_te)])

plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Simulated Dataset")

plt.show()

plt.scatter(y_te, y_pred_gd)
plt.scatter(y_te, y_pred_ne)
plt.scatter(y_te, y_pred_sk)

plt.plot([min(y_te), max(y_te)],
         [min(y_te), max(y_te)])

plt.xlabel("Actual Chance of Admit")
plt.ylabel("Predicted Chance of Admit")
plt.title("Admissions Dataset")

plt.show()
