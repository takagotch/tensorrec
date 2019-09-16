### tensorrec
---
https://github.com/jfkirk/tensorrec

```py
// test/test_tensorrec.py

class TensorRecTestCase(TestCase):

  @classmethod
  def setUpClass(cls):
  
    cls.n_user_features = 200
    cls.n_item_features = 150
    
    cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
      num_users=15, num_items=30, interaction_density=.5, num_user_features=cls.n_user_features,
      num_item_features=cls.n_item_features, n_features_per_user=20, n_features_per_item=20, pos_int_ration=5
    )
    
    set_session(None)
    cls.temp_dir = tempfile.mkdtemp()
    cls.interactions_path = os.path.join(cls.temp_dir, 'interactions.tfrecord')
    cls.user_features_path = os.path.join(cls.temp_dir, 'user_features.tfrecord')
    cls.item_features_path = os.path.join(cls.temp_dir, 'item_features.tfrecord')

  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.temp_dir)
  
  def test_init():
    self.assertIsNotNone(TensorRec())
  
  def test_init_fail_0_components(self):
    with self.assertRaises(ValueError):
      TensorRec(n_components=0)
  
  def test_init_fail_none_factory(self):
  
  
  
  def test_init_fail_bad_loss_graph(self):
  
  
  def test_init_fail_attension_with_1_taste(self):
  
  
  def test_init_fail_bad_attension_graph(self):
  
  
  def test_predict_fail_unfit(self):
  
  def test_fit_verbose(self):
  
  
  def test_fit_batched(self):
  
  det test_fit_fail_bad_input(self):
  
  def test_fit_mismatched_batches(self):
  
  
  def test_fit_fail_batching_dataset(self):
  
  def test_fit_user_features_as_dataset(self):
  
  def test_fit_item_feature_as_dataset(self):
  
  def test_fit_item_feature_as_dataset(self):
  
  def test_fit_interactions_as_dataset(self):
  
  def test_fit_interactions_as_dataset(self):
  
  def test_from_datasets(self):
    uf_as_dataset = create_tensorrec_dataset_from_sparse_matrix()
    if_as_dataset = create_tensorrect_dataset_from_sparse_matrix()
    int_as_dataset = create_tensorrect_data_from_sparse_matrix()
    model = TensorRec(n_components=10)
    model.fit(int_as_dataset, uf_as_dataset, if_as_dataset, epochs=10)
    
  def test_fit_from_tfrecords(self):
    set_session(None)
    model = TensorRec(n_components=10)
    model.fit(self.interactions_path, self.user_features_path, self.item_features_path, epochs=10)

class TEnsorRecAPITestCase(TestCase):
  
  @classmethod
  def setUpClass(cls):

    cls.n_users = 15
    cls.n_items = 30
    
    cls.iteractions, cls.user_features, cls.item_features = generate_dummy_data(
      num_users=cls.n_users, num_items=cls.n_items, interaction_density=.5, num_user_features=200,
      num_item_features=200, n_features_per_user=20, n_features_per_item=20, pos_int_ratio-.5
    )
    
    cls.standard_model = TensorRect()
    cls.standard_model.fit()
    
    cls.unbiased_model = TensorRec(n_components=10)
    cls.standard_model.fit(cls.iteractions, cs.user_features, cls.item_features, epochs=10)
    
  def test_fit(self):
    self.assertIsNotNone(self.standard_model.tf_prediction)
    
  def test_predict(self):
    predictions = self.standard_model.predict(user_features=self.user_features,
        item_features=self.item_features)
        
    self.assertEqual(predictions.shape, (self.n_users, self.n_items))
  
  def test_predict_rank(self):
    ranks = self.standard_model.predict_rank(user_features=self.user_features,
        item_features=self.item_features)
    self.assertEqual(ranks.shape, (self.n_users, self.n_items))
    for x in range(ranks.shape[0]):
      for y in range(ranks.shape[1]):
        val = ranks[x][y]
        self.assertGreater(val, 0)
  
  def test_predict_similar_items(self):
    sims = self.standard_model.predict_similar_items(item_features=self.item_features,
        item_ids=[6, 12],
        n_similar=5)
    
    self.assertEqual(len(sims), 2)
    
    for item_sims in sims:
      self.assertEqual(len(item_sims), 5)
      
  def test_fit_predict_unbiased(self):
  
  def test_predict_user_repr(self):
  
  def test_predct_items_repr(self):
  
  def test_predict_user_bias_unbiases_model(self):
  
  def test_predict_item_bias_unbiased_model(self):
  
  def test_predict_user_attn_repr(self):
  
class TensorRecBaiasePrediction(TestCase):
  
  @classmethod
  def setUpClass(cls):
    cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
      num_users=15, num_items=30, interaction_density=.5, num_user_features=200, num_item_features=200,
      n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
    )
    
    cls.standard_model = TensorRec(n_components=10)
    cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, eopchs=10)
    
  def test_predict_user_bias(self):
  
  def test_predict_item_bias(self):
  
class TensorRecAPINTastesTestCase(TensorRecAPITestCase):
  
  @classmethod
  def setUpClass(cls):
  
    cls.n_users = 15
    cls.n_items = 30
    
    cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
    )
    
    cls.standard_model = TensorRec()
    cls.standard_model.fit()
    cls.unbiased_model = TensorRec()
    cls.unbiased_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)
    
  def test_predict_user_repr(self):
    user_repr = self.unbiased_model.predict_user_representation(self.user_features)
    
    self.assertEqual(user_repr.shape, (3, self.user_features.shape[0], 10))

class TensorRecAPIAttentionTestCase(TesnsorRecAPINTastesTestCase):

  @classmethod
  def setUpClass(cls):
  
    cls.n_users = 15
    cls.n_items = 30
    
    cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
    )
    
    cls.standard_model = TensorRec()
    cls.standard_model.fit()
    cls.unbiased_model.fit()
    cls.unbiased_model.fit()
    
  def test_predict_user_attn_repr(self):
  
class TensorRecAPIDatasetInputTestCase(TesnsorRecAPITestCase):

  @classmethod
  def setUpClass(cls):
   
    cls.n_users = 15
    cls.n_items = 30
    
    cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
    )
    
    cls.standard_model = TensorRec(n_components=10)
    cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)
    
    cls.unbiased_model = TensorRec()
    cls.unbiased_model.fit()
    
class TensorRecAPIRecordInputTestCase(TensorRecAPITestCase):

  @classmethod
  def setUpClass(cls):
  
    set_session(None)
    
    cls.n_users = 15
    cls.n_items = 30
    
    int_ds, uf_ds, if_ds = generate_dummy_data(
    )
    
    cls.temp_dir = tempfile.mkdtemp()
    cls.interactions = os.path.join(cls.temp_dir, 'interactions.tfrecord')
    cls.user_features = os.path.join(cls.temp_dir, 'user_features.tfrecord')
    cls.item_featuers = os.path.join()
    
    write_tfrecord_from_sparse_matrix()
    write_tfrecord_from_sparse_matric()
    write_tfrecord_from_sparse_matrix()
    
    cls.standard_model = TensorRec()
    cls.standard_model.fit()
    
    cls.unbiased_model = TensorRec()
    cls.unbiased_model.fit()
    
  @classmethod
  def tearDownClass():
    shutil.rmtree()
    
class TensorRecSavingTestCase(TestCase):
  
  @classmethod
  def setUpClass():
    cls.interactions, cls.user_features, cls.items_features = generate_dummy_data(
    )
    tf.reset_default_graph()
    set_session(None)
    
  def setUp(self):
    self.test_dir = tempfile.mkdtemp()
    
  def tearDown(self):
  
  def test_save_and_load_model(self):
  
  def test_save_and_load_model_same_session(self):
    


```

```
```

```
```


