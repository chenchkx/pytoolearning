import simbench as sb

grid_code = "1-HV-urban--0-sw"
net = sb.get_simbench_net(grid_code)

profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)

sgen_p = profiles[("sgen", "p_mw")]
load_p = profiles[("load", "p_mw")]
load_q = profiles[("load", "q_mvar")]

import pandas as pd
# X = pd.concat([sgen_p, load_p, load_q], axis=1)
X = pd.read_json("./res_bus/vm_pu.json")
y = pd.read_json("./res_line/loading_percent.json")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X ,y, train_size=0.1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train= scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

from sklearn.neural_network import MLPRegressor
ann = MLPRegressor(verbose=1)
ann.fit(X_train,y_train)
y_predict = ann.predict(X_test)
# y_predict = scaler.inverse_transform(y_predict)

import matplotlib.pyplot as plt
plt.plot(y_test[3000:3100,99],linestyle="--",label="cl")
plt.plot(y_predict[3000:3100,99],linestyle="-",label="pl")
plt.legend()
plt.show()





print('testing')