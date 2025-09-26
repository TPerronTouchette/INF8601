#include <stdio.h>

#include "filter.h"
#include "pipeline.h"
#include "queue.h"

const int MAX_QUEUE_SIZE = 10;

queue_t* image_loaded_queue;
queue_t* image_scaled_queue;
queue_t* image_sharpenned_queue;
queue_t* image_sobelled_queue;

bool image_loader_finished;
bool image_scaler_finished;
bool image_sharpenner_finished;
bool image_sobeller_finished;

void *image_loader(void *arg) {
	image_dir_t *image_dir = (image_dir_t *) arg;
	while (1) {
		image_t* image = image_dir_load_next(image_dir);
        if (image == NULL) {
			printf("loader has finish!\n");
            break;
        }

		queue_push(image_loaded_queue, image);
	}

	image_loader_finished = true;
	return 0;
}

void *image_scaler(void *arg) {
	while (1) {
		if (image_loaded_queue->used == 0 && image_loader_finished) {
			printf("scaler has finish!\n");
            break;
		}
		image_t* image = queue_pop(image_loaded_queue);
        if (image == NULL) continue;

		queue_push(image_scaled_queue, filter_scale_up(image, 2));
		image_destroy(image);
	}

	image_scaler_finished = true;
	return 0;
}

void *image_sharpenner(void *arg) {
	while (1) {
		if (image_scaled_queue->used == 0 && image_scaler_finished) {
			printf("sharpenner has finish!\n");
            break;
		}
		image_t* image = queue_pop(image_scaled_queue);
        if (image == NULL) continue;

		queue_push(image_sharpenned_queue, filter_sharpen(image));
		image_destroy(image);
	}
	
	image_scaler_finished = true;
	return 0;
}

void *image_sobeller(void *arg) {
	while (1) {
		if (image_sharpenned_queue->used == 0 && image_sharpenner_finished) {
			printf("sobeller has finish!\n");
            break;
		}
		image_t* image = queue_pop(image_sharpenned_queue);
        if (image == NULL) continue;

		queue_push(image_sobelled_queue, filter_sobel(image));
		image_destroy(image);
	}
	
	image_sobeller_finished = true;
	return 0;
}

void *image_saver(void *arg) {
	image_dir_t *image_dir = (image_dir_t *) arg;
	while (1) {
		if (image_sobelled_queue->used == 0 && image_sobeller_finished) {
			printf("saver has finish!\n");
            break;
		}
		image_t* image = queue_pop(image_sobelled_queue);
        if (image == NULL) continue;

		image_dir_save(image_dir, image);
		printf(".");
		fflush(stdout);
		image_destroy(image);
	}
	return 0;
}

int pipeline_pthread(image_dir_t* image_dir) {
	pthread_t threads[5];

	image_loaded_queue = queue_create(MAX_QUEUE_SIZE);
	image_scaled_queue = queue_create(MAX_QUEUE_SIZE);
	image_sharpenned_queue = queue_create(MAX_QUEUE_SIZE);
	image_sobelled_queue = queue_create(MAX_QUEUE_SIZE);

	image_loader_finished = false;
	image_scaler_finished = false;
	image_sharpenner_finished = false;
	image_sobeller_finished = false;

    
	pthread_create(&threads[0], NULL, image_loader, image_dir);
	pthread_create(&threads[1], NULL, image_scaler, NULL);
    pthread_create(&threads[2], NULL, image_sharpenner, NULL);
	pthread_create(&threads[3], NULL, image_sobeller, NULL);
	pthread_create(&threads[4], NULL, image_saver, image_dir);
	for (int i = 0; i < 5; ++i) {
		pthread_join(threads[i], NULL);
	}
	printf("end\n");
	queue_destroy(image_loaded_queue);
	queue_destroy(image_scaled_queue);
	queue_destroy(image_sharpenned_queue);
	queue_destroy(image_sobelled_queue);

	printf("\n");
	return 0;
}
