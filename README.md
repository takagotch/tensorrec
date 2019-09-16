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
  
  def test_init(self):
    self.assertIsNotNone(TensorRec())
  
  def test_init_fail_0_components(self):
    with self.assertRaises(ValueError):
      TensorRec(n_components=0)
  
  def test_init_fail_none_factory(self):
    with self.assertRaises(ValueError):
      TensorRec(user_repr_graph=None)
    with self.assertRaises(valueError):
      TensorRec(item_repr_graph=None)
    with self.assertRaised(ValueError):
      TensorRec(loss_graph=None)
  
  def test_init_fail_bad_loss_graph(self):
    with self.assertRaises(ValueError):
      TensorRec(loss_graph=np.mean)
  
  def test_init_fail_attension_with_1_taste(self):
    with self.assertRaises(ValueError):
      TensorRec(n_tastes=1, attention_graph=LinearRepresentationGraph())
  
  def test_init_fail_bad_attension_graph(self):
    with self.assertRaises(ValueError):
      TensorRec(attention_graph=np.mean)
  
  def test_predict_fail_unfit(self):
    model = TensorRec()
    with self.assertRaises(ModelNotFitExceptoin):
      model.predict(self.user_features, self.item_features)
    with self.assertRaises(ModelNotFitException):
      model.predict_rank(self.user_features, self.item_features)
    
    with self.assertRaises(ModelNotFitException):
      modelpredict_user_representation(self.user_features)
    with self.assertRaises(ModelNotFitException):
      model.predict_item_representation(self.item_features)
    with self.assertRaises(ModelNotFitException):
      model.predict_user_attention_representation(self.user_features)
  
    with self.assertRaises(ModelNotFitException):
      model.predict_similar_items(self.item_features, item_ids=[1], n_similar=5)
 
    with self.assertRaises(ModelNotFitException):
      model.predict_item_bias(self.item_features)
    with self.assertRaises(ModelNotFitException):  
      model.predict_user_bias(self.user_features)
  
  def test_fit_verbose(self):
    model = TensorRec(n_components=10)
    model.fit(self.intractions, self.user_features, self.item_features, epochs=10, verbose=True)
  
    self.assertIsNotNone(model.tf_prediction)
  
  def test_fit_batched(self):
    model = TensorRec(n_components=10)
    model.fit(self.interactions, self.user_features, self.item_features, epochs=10, user_batch_size=2)
  
    self.assertIsNotNone(model.tf_prediction)
    
  def test_fit_batched(self):
    model = TensorRec(n_components=10)
    model.fit(self.interactions, self.user_features, self.item_features, epochs=10, user_batch_size=2)
    
    self.assertIsNotNone(model.tf_prediction)
    
  def test_fit_fail_bad_input(self):
    model = TensorRec(n_components=10)
    with self.assertRaises(ValueError):
      model.fit(np.array([1, 2, 3, 4]), self.user_features, self.item_features, epochs=10)
    with self.assertRaises(ValueError):
      model.fit(self.interactions, np.array([1, 2, 3, 4]), self.item_features, epochs=10)
    with self.assertRaised(ValueError):
      model.fit(self.interactions, self.user_features, np.array([1, 2, 3, 4]), epochs=10)
    
  def test_fit_fail_mismatched_batches(self):
    model = TensorRec(n_components=10)
    with self.assertRaises(ValueError):
      model.fit(self.interactions,
        [self.user_features, self.user_features],
        [self.item_features, self.item_features, self.item_features],
        epochs=10)
        
    with self.assertRaises(ValueError):
      model.fit(self.interactions,
        [self.user_features, self.user_features],
        [self.item_features, self.item_features],
        epochs=10)
    
    model.fit([self.interactions, self.interactions],  
      [self.user_features, self.user_features],
      self.item_features,
      epoch=10)
      
    model.fit([self.interactions, self.interactions],
      [self.user_features, self.user_features],
      [self.item_featuers, self.item_features],
      epochs=10)
  
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
    
    cls.standard_model = TensorRect(n_components=10)
    cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)
    
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
    predictions = self.unbiased_model.predict(user_features=self.user_features, item_features=self.item_features)
    self.assertEqual(predictions.shape, (self.n_user, self.n_items))
  
  def test_predict_user_repr(self):
    user_repr = self.unbiased_model.predict_user(self.user_features)
    self.assertEqual(user_repr.shape, (self.n_users, 10))
  
  def test_predct_items_repr(self):
    item_repr = self.unbiassed_model.predict_item_representation(self.item_features)
    self.assertEqual(item_repr.shap, (self.n_items, 10))
  
  def test_predict_user_bias_unbiases_model(self):
    self.assertRaises(
      ModelNotBiasedException,
      self.unbiased_model.predict_user_bias,
      self.user_features)
  
  def test_predict_item_bias_unbiased_model(self):
    self.assertRaises(
      ModelNotBiasedException,
      self.unbiased_model.predict_item_bias,
      self.item_features)
  
  def test_predict_user_attn_repr(self):
    with self.assertRaises(ModelWithoutAttentionException):
      self.unbiased_model.predict_user_attention_representation(self.user_features)
  
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
    user_bias = self.standard_model.predict_user_bias(self.user_features)
    self.assertTrue(any(user_bias))
  
  def test_predict_item_bias(self):
    item_bias = self.standard_model.predict_item_bias(self.item_features)
    self.asserTrue(any(item_bias))
  
class TensorRecAPINTastesTestCase(TensorRecAPITestCase):
  
  @classmethod
  def setUpClass(cls):
  
    cls.n_users = 15
    cls.n_items = 30
    
    cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
      num_users=cls.n_users, num_items=cls.n_items, interaction_density=.5, num_user_features=200,
      num_item_features=200, n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
    )
    
    cls.standard_model = TensorRec(n_components=10,
        n_tastes=3,
        user_repr_graph=NormalizedLinearRepresentationGraph())
    cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)
    cls.unbiased_model = TensorRec(n_components=10,
        n_tastes=3,
        biased=False,
        user_repr_graph=NormalizedLinearRepresentationGraph())
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
      num_users=cls.n_users, num_items=cls.n_items, interaction_density=.5, num_user_features=200,
      num_item_features=200, n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
    )
    
    cls.standard_model = TensorRec(n_components=10,
        n_tastes=3,
        user_repr_graph=NormalizedLinearLinearRepresentationGraph(),
        attention_graph=LinearRepresentationGraph())
    cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)
    cls.unbiased_model.fit(n_components=10,
        n_tastes=3,
        biased=False,
        user_repr_graph=NormalizedLinearRepressentationGraph(),
        attention_graph=LinearRepresentationGraph())
    cls.unbiased_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)
    
  def test_predict_user_attn_repr(self):
    user_attn_repr = self.unbiased_model.predict_user_attention_representation(self.user_features)
    
    self.assertEqual(user_attn_repr.shape, (3, self.user_features.shape[0], 10))
  
class TensorRecAPIDatasetInputTestCase(TesnsorRecAPITestCase):

  @classmethod
  def setUpClass(cls):
   
    cls.n_users = 15
    cls.n_items = 30
    
    cls.interactions, cls.user_features, cls.item_features = generate_dummy_data(
      num_users=cls.n_users, num_items=cls.n_items, interaction_density=.5, num_user_features=200,
      num_item_features=200, n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5,
      return_datasets=True
    )
    
    cls.standard_model = TensorRec(n_components=10)
    cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)
    
    cls.unbiased_model = TensorRec(n_components=10, biased=False)
    cls.unbiased_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)
    
class TensorRecAPIRecordInputTestCase(TensorRecAPITestCase):

  @classmethod
  def setUpClass(cls):
  
    set_session(None)
    
    cls.n_users = 15
    cls.n_items = 30
    
    int_ds, uf_ds, if_ds = generate_dummy_data(
      num_users=cls.n_users, num_items=cls.n_items, interaction_density=.5, num_user_features=200,
      num_item_features=200, n_features_per_user=20, n_features_per_item=20, pos_int_ratio=.5
    )
    
    cls.temp_dir = tempfile.mkdtemp()
    cls.interactions = os.path.join(cls.temp_dir, 'interactions.tfrecord')
    cls.user_features = os.path.join(cls.temp_dir, 'user_features.tfrecord')
    cls.item_featuers = os.path.join(cls.temp_dir, 'item_features.tfrecord')
    
    write_tfrecord_from_sparse_matrix(cls.interactions, int_ds)
    write_tfrecord_from_sparse_matric(cls.user_features, uf_ds)
    write_tfrecord_from_sparse_matrix(cls.item_features, if_ds)
    
    cls.standard_model = TensorRec(n_components=10)
    cls.standard_model.fit(cls.interactions, cls.user_features, cls.item_features, epochs=10)
    
    cls.unbiased_model = TensorRec(n_componetns=10, biased=False)
    cls.unbiased_model.fit(cls.interactions, cls.user_fatures, cls.item_features, epochs=10)
    
  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.temp_dir)
    
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
    shutilrmtree(self.test_dir)
  
  def test_save_and_load_model(self):
    model = TEnsorRect(n_components=10)
    model.fit(self.interactions, self.userr_features, self.item_features, epochs=10)
    
    predictions = model.predict(user_features=self.user_features, item_features=self.item_features)
    ranks = model.predict_rank(user_features=self.user_features, item_features=self.item_features)
    model.save_model(directory_path=self.test_dir)
    
    predictions_after_save = model.predict(user_featuers=self.user_features, item_features, item_features=self.item_features)
    ranks_after_save = model.predict_rank(user_features=self.user_features, item_features=self.item_features)
    self.assertTrue((predictions == predictions_after_save).all())
    self.assertTrue((rank == ranks_after_save).all())
    
    set_session(None)
    tf.reset_default_graph()
    
    new_model = TensorRec.load_model(directory_path=self.test_dir)
    new_predictions = new_model.predict(user_features=self.user_features, item_features=self.item_features)
    new_ranks = new_model.predict_rank(user_features=self_features, item_features=self.item_features)
    
    self.assertTrue((predictions == new_predictions).all())
    self.assertTrue((ranks == new_ranks).all())
  
  def test_save_and_load_model_same_session(self):
    model = TensorRec(n_components=10)
    model.fit(self.interactions, self.user_features, self.item_features, epochs=10)
    
    predictions = model.predict(user_features=self.user_features, item_features=self.item_features)
    ranks = model.predict_rank(user_features=self.user_features, item_features=self.item_features)
    model.save_model(directory_path=self.test_dir)
    
    new_model = TensorRec.load_model(directory_path=self.test_dir)
    new_predictions = new_model.predict(user_features=self.user_features, item_features=self.item_features)
    new_ranks = new_model.predict_rank(user_features=self.user_features, item_features=self.item_features)
    
    self.assertTrue((predictions == new_predictions).all())
    self.assertTrue((ranks == new_ranks).all())


```

```
```

```
```


