import joblib
m=joblib.load('mental/trained_model.joblib')
print('type:', type(m))
print('n_outputs_: ', getattr(m,'n_outputs_', None))
print('n_features_in_: ', getattr(m,'n_features_in_', None))
try:
    print('estimators_ len:', len(m.estimators_))
except Exception as e:
    print('estimators_ error', e)
try:
    print('estimator_.n_outputs_: ', getattr(m.estimator_, 'n_outputs_', None))
except Exception as e:
    print('estimator_ error', e)
print('done')
