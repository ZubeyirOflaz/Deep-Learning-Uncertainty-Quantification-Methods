casting_args = {
    'train_set_path': "N:\\Thesis\\modified\\train\\",
    'test_set_path': "N:\\Thesis\\modified\\test\\"

}
dataset_paths = {
    'arrhythmia_train': '\\Datasets\\arrhythmia\\arrhythmia_train.npy',
    'arrhythmia_test': '\\Datasets\\arrhythmia\\arrhythmia_test.npy',
    'casting_train': '\\Datasets\\casting_modified\\train\\',
    'casting_test': '\\Datasets\\casting_modified\\test\\',
    'ford_a_train': '\\Datasets\\ford_a\\FordA_TRAIN.arff',
    'ford_a_test': '\\Datasets\\ford_a\\FordA_TEST.arff'
}
models = {
    'base_models': {
        'arrhythmia': 'Models\\arrhythmia_base.pickle',
        'casting' : 'Models\\casting_base.pickle',
        'ford_a' : '\\Models\\ford_a_base.pickle'
    },
    'laplace_approximation':{
      'casting' : '\\Models\\laplace_approximation_casting.pkl',
      'arrhythmia' : '\\Models\\laplace_approximation_arrhythmia.pickle'
    },
    'mimo_models' : {
        'casting_model' : 'Models\\casting_mimo_model.pyt',
        'casting_study' : 'Models\\casting_mimo_study.pkl'
    }
}