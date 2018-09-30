import tensorflow as tf


def build_model(features: tf.Tensor,
                target_images: tf.Tensor,
                mode: int,
                network_fn,
                loss_fn,
                input_image_fn,
                learning_rate) -> tf.estimator.EstimatorSpec:
    """
    Common functionality for different denoising models.
    Initializes the layers using provided network_fn function
    and builds the necessary EstimatorSpecs for
    the specified `mode`.
    :param features: The tensor representing input images
    :param target_images: The tensor representing target images
    :param mode: current estimator mode; should be one of
           tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    :param network_fn: The function that builds and return a deep neural network
    :param loss_fn: The function that builds all loss components via tf.add_loss
    :param input_image_fn: The function that returns original input image
           from the provided features
    :param learning_rate: Floating point value or a function that returns
           the current learning rate given the current global_step
    :return: EstimatorSpec parameterized according to the input params
             and the current mode.
    """

    input_images = input_image_fn(features)
    output_images = network_fn(features)

    predictions = {
        'input_images': input_images,
        'output_images': output_images
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    tf.summary.image("inputs", input_images)
    tf.summary.image("outputs", output_images)
    tf.summary.image("targets", target_images)

    # Calculate loss
    loss_fn(output_images, target_images)
    loss = tf.losses.get_total_loss(add_regularization_losses=True)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        if callable(learning_rate):
            learning_rate_tensor = learning_rate(global_step)
        else:
            learning_rate_tensor = tf.convert_to_tensor(
                learning_rate, name='learning_rate')

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        tf.summary.scalar('learning_rate', learning_rate_tensor)

        train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)


def create_estimator(options, derived_model_fn, dataset_size, is_training):
    run_config = None
    estimator_params = {
        'batch_size': 1,
        'num_images': 1,
        'num_epochs': 1
    }

    if is_training:
        # Set up a RunConfig to save checkpoint for training.
        run_config = tf.estimator.RunConfig().replace(
            save_checkpoints_secs=300,
            keep_checkpoint_max=100)
        estimator_params = {
            'batch_size': options.batch_size,
            'num_images': dataset_size,
            'num_epochs': options.train_epochs
        }

    return tf.estimator.Estimator(
        model_fn=derived_model_fn,
        model_dir=options.model_dir,
        config=run_config,
        params=estimator_params)
