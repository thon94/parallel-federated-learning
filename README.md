# Parallelized TensorFlow Federated Implementation

This repository demonstrates an example of using TensorFlow v1 for Federated Learning. The training phase is accelerated by parallelizing the static graph. Results show a significant improvement in reducing training time.

First, define a number of parallel "slots" (models) to be trained simultaneously. All TenforFlow (tf) operations are applied on all slots in the same manner. The operations/variables are distinguished across slots by using variable scope.