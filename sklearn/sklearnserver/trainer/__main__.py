
# import numpy as np
# from .model import Model


# # Load train & valid data
# train_num = 200
# train_x = np.random.rand(train_num, 10)
# train_y = np.random.randint(2, size=(train_num, 1))

# print(train_x.shape, train_y.shape)

# eval_num = 50
# eval_x = np.random.rand(eval_num, 10)
# eval_y = np.random.randint(2, size=(eval_num, 1))

# print(eval_x.shape, eval_y.shape)


# # Create a Model
# INPUT_DIM = train_x.shape[1:]  # (10, )
# model = Model()


# # Create Datasets
# NUM_EPOCHS = 10
# BATCH_SIZE = 16

# # Training
# model.train(train_x, train_y)


# # Evaluation
# eval_res = model.evaluate(eval_x, eval_y)

# # Save
# model.save('./0001', compress=1)


# # Load & Reuse
# model2 = Model(filepath='./0001')

# eval_res2 = model2.evaluate(eval_x, eval_y)
# print(eval_res)

# # model2.train(train_x, train_y)

# model2.predict(np.random.rand(1, 10))


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0)
pipe1 = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe1.fit(X_train, y_train)

test1 = pipe1.score(X_test, y_test)

import joblib
joblib.dump(pipe1, 'model.joblib')

pipe2 = joblib.load('model.joblib')
test2 = pipe2.score(X_test, y_test)

print(test1)
print(test2)
