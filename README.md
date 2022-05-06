
### Mehmet Ali Nebioglu


[Recommender Systems](https://github.com/malinphy/recommender_sys)
Deep learning based modern recommendation systems has been worked. 
- [Feed Forward Neural Network with ranking](https://github.com/malinphy/recommender_sys/tree/main/YouTube/anime_dataset/dataprocess)
- Convolutional Ranking based recommender system
  -```
    class RankingModel(tf.keras.Model):

      def __init__(self):
        super().__init__()
        embedding_dimension = 32

    # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

    # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
          vocabulary=unique_movie_titles, mask_token=None),
        tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])

    # Compute predictions.
        self.ratings = tf.keras.Sequential([
        # Learn multiple dense layers.
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
    # Make rating predictions in the final layer.
        tf.keras.layers.Dense(1)
        ])
    
    def call(self, inputs):

      user_id, movie_title = inputs

      user_embedding = self.user_embeddings(user_id)
      movie_embedding = self.movie_embeddings(movie_title)

      return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))

    def build_graph(self):
      x = [Input(shape=(1)),
         Input(shape=(1))
         ]
      return Model(inputs=x, outputs=self.call(x))


    plot_model(
          RankingModel().build_graph(), 
          show_shapes=True,
          # show_layer_names=True,
          rankdir='TB',
          expand_nested=True,
      )
-```
- [BERT4Rec](https://github.com/malinphy/recommender_sys/tree/main/BERT4Rec)
  - ![ranking](https://user-images.githubusercontent.com/55249305/167198809-d8d3ab5f-f414-4993-8911-c8624c7fae3e.png)


[Natural Language Processing](https://github.com/malinphy/Embedding_calls)
