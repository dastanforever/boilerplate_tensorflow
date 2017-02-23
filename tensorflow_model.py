import numpy as np
import tensorflow as tf

from config import Config
from utils.general_utils import Progbar

class NetworkModel:
	"""This is the actual class where you can implement the
		neural network model, an several layers, along with
		placeholders, constants, weights and other things.
	"""
	def __init__(self, config, **kw):
		self.config = config

		for (key, value) in kw:
			self.args[key] = value

		# building the model.
		self.add_placeholders()
		self.pred = self.add_prediction_op()
		self.loss = self.add_loss_op(self.pred)
		self.train_op = self.add_training_op(self.loss)

	def add_placeholders(self):
		# self.placeholder1 = tf.placeholder()
		# self.placeholder2 = tf.placeholder()

		raise NotImplementedError("Add placeholders.")

	def create_feed_dict(self):
		# Create the feed dict. It would contain all the key
		# value pairs.
		
		raise NotImplementedError("Add Feed Dictionary.")

	def add_prediction_op(self):
		# Write the code for prediction operation here.
		# Declare the variables.

		raise NotImplementedError("Add prediction operations.")
		# return prediction

	def add_loss_op(self):
		# Write the loss function here. Use the prediction
		# returned from the add_prediction_op function.

		raise NotImplementedError("Add Loss function.")
		# return loss

	def add_training_op(self):
		# Add the optimizer here.

		raise NotImplementedError("Add training optimizer.")
		# return training_optimizer

	def train_on_batch(self, sess, inputs_batch, labels_batch):
		"""
		Prototype for training on batch. This also defines the prototype
		for feed dict.
		feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, 
					dropout=self.config.dropout)
		_, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
		return loss
		"""

		raise NotImplementedError("Implement batch loss code.")
		# return loss

	def run_epoch(self, sess, parser, train_examples, dev_set):
		prog = Progbar(target=1 + len(train_examples) / self.config.batch_size)
		for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
			loss = self.train_on_batch(sess, train_x, train_y)
			prog.update(i + 1, [("train loss", loss)])

		# Print evaluation results on dev set.

		# print "Evaluating on dev set",
		# dev_UAS, _ = parser.parse(dev_set)
		# print "- dev UAS: {:.2f}".format(dev_UAS * 100.0)
		# return dev_UAS
		raise NotImplementedError("Implement evaluation on training set, and return dev_set loss.")


	def fit(self, sess, saver, parser, train_examples, dev_set):
		best_dev_score = 0
		for epoch in range(self.config.n_epochs):
			print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
			dev_score = self.run_epoch(sess, parser, train_examples, dev_set)
			if dev_score > best_dev_score:
				best_dev_score = dev_score
				if saver:
					# Store the training weight matrix.
					print("New best dev Score! Saving model in ./data/weights/parser.weights")
					saver.save(sess, './data/weights/parser.weights')
			print

def main(debug=True):
	print(80 * "=")
	print("INITIALIZING")
	print(80 * "=")

	config = Config()
    # load preprocessed data.
    # prototype.
    # parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
	
	# Directory to store learned weights,
	if not os.path.exists('./data/weights/'):
		os.makedirs('./data/weights/')

	with tf.Graph().as_default():
		print("Building model...",
		start = time.time())

		# sample data -- embeddings.
		model = NetworkModel(config, embeddings=embeddings)
		print("took {:.2f} seconds\n".format(time.time() - start))

		init = tf.global_variables_initializer()
		# If you are using an old version of TensorFlow, you may have to use
		# this initializer instead.
		# init = tf.initialize_all_variables()
		saver = None if debug else tf.train.Saver()

		 with tf.Session() as session:
		 	# the predictor class here is parser.
			parser.session = session
			session.run(init)

			print(80 * "=")
			print("TRAINING")
			print(80 * "=")
			model.fit(session, saver, parser, train_examples, dev_set)

			if not debug:
				print(80 * "=")
				print("TESTING")
				print(80 * "=")
				print("Restoring the best model weights found on the dev set")
				saver.restore(session, './data/weights/parser.weights')

				print("Final evaluation on test set",
				UAS, dependencies = parser.parse(test_set))

				print("- test UAS: {:.2f}".format(UAS * 100.0))
				print("Writing predictions")
				# Dump results.
				with open('q2_test.predicted.pkl', 'w') as f:
					cPickle.dump(dependencies, f, -1)

				print("Done!")

if __name__ == '__main__':
    main()

