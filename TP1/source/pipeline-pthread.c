#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_QUEUE_SIZE 10
#define MIN_NB_THREAD 4

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdatomic.h>

#include "filter.h"
#include "pipeline.h"
#include "queue.h"

queue_t* image_loaded_queue;
queue_t* image_scaled_queue;
queue_t* image_sharpenned_queue;
queue_t* image_sobelled_queue;

_Atomic int image_loader_running;
_Atomic int image_scaler_running;
_Atomic int image_sharpenner_running;
_Atomic int image_sobeller_running;
_Atomic int image_saver_running;

void *image_loader(void *arg) {
	image_dir_t *image_dir = (image_dir_t *) arg;
	while (1) {
		image_t* image = image_dir_load_next(image_dir);
        if (image == NULL) break;

		queue_push(image_loaded_queue, image);
	}
	atomic_fetch_sub(&image_loader_running, 1);

	if (atomic_load(&image_loader_running) == 0) {
		int j = atomic_load(&image_scaler_running);
		for (int i = 0; i < j; ++i) queue_push(image_loaded_queue, NULL);
	}
	
	return 0;
}

void *image_scaler(void *arg) {
	while (1) {
		image_t* image = queue_pop(image_loaded_queue);
        if (image == NULL && atomic_load(&image_loader_running) == 0) break;
		else if (image == NULL) continue;

		queue_push(image_scaled_queue, filter_scale_up(image, 2));
		image_destroy(image);
	}

	atomic_fetch_sub(&image_scaler_running, 1);

	if (atomic_load(&image_scaler_running) == 0) {
		int j = atomic_load(&image_sharpenner_running);
		for (int i = 0; i < j; ++i) queue_push(image_scaled_queue, NULL);
	}

	return 0;
}

void *image_sharpenner(void *arg) {
	while (1) {
		image_t* image = queue_pop(image_scaled_queue);
        if (image == NULL && atomic_load(&image_scaler_running) == 0)	break;
		else if (image == NULL) continue;

		queue_push(image_sharpenned_queue, filter_sharpen(image));
		image_destroy(image);
	}

	atomic_fetch_sub(&image_sharpenner_running, 1);

	if (atomic_load(&image_sharpenner_running) == 0) {
		int j = atomic_load(&image_sobeller_running);
		for (int i = 0; i < j; ++i) queue_push(image_sharpenned_queue, NULL);
	}

	return 0;
}

void *image_sobeller(void *arg) {
	while (1) {
		image_t* image = queue_pop(image_sharpenned_queue);
        if (image == NULL && atomic_load(&image_sharpenner_running) == 0) break;
		else if (image == NULL) continue;
		
		queue_push(image_sobelled_queue, filter_sobel(image));
		image_destroy(image);
	}

	atomic_fetch_sub(&image_sobeller_running, 1);

	if (atomic_load(&image_sobeller_running) == 0) {
		int j = atomic_load(&image_saver_running);
		for (int i = 0; i < j; ++i) queue_push(image_sobelled_queue, NULL);
	}

	return 0;
}

void *image_saver(void *arg) {
	image_dir_t *image_dir = (image_dir_t *) arg;
	while (1) {
		image_t* image = queue_pop(image_sobelled_queue);
		if (image == NULL && atomic_load(&image_sobeller_running) == 0) break;
		else if (image == NULL) continue;

		image_dir_save(image_dir, image);
		printf(".");
		fflush(stdout);
		image_destroy(image);
	}

	atomic_fetch_sub(&image_saver_running, 1);

	return 0;
}

int pipeline_pthread(image_dir_t* image_dir) {
	long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
	unsigned int nb_threads = MAX(MIN_NB_THREAD, nprocs);
	pthread_t thread_loader;
	pthread_t *threads = malloc(nb_threads * sizeof(pthread_t));

	void *(*tasks[MIN_NB_THREAD])(void*) = {image_scaler, image_sharpenner, image_sobeller, image_saver};
	_Atomic int *atomic_values[MIN_NB_THREAD] = {&image_scaler_running, &image_sharpenner_running, &image_sobeller_running, &image_saver_running};
	image_loaded_queue = queue_create(MAX_QUEUE_SIZE);
	image_scaled_queue = queue_create(MAX_QUEUE_SIZE);
	image_sharpenned_queue = queue_create(MAX_QUEUE_SIZE);
	image_sobelled_queue = queue_create(MAX_QUEUE_SIZE);
	
	atomic_fetch_add(&image_loader_running, 1);
	pthread_create(&thread_loader, NULL, image_loader, image_dir);

    int ID_task = 0;
	for (int i = 0; i < nb_threads; ++i) {
		ID_task = i % MIN_NB_THREAD;
		atomic_fetch_add(atomic_values[ID_task], 1);
		pthread_create(&threads[i], NULL, tasks[ID_task], image_dir);
	}

	pthread_join(thread_loader, NULL);
	for (int i = 0; i < nb_threads; ++i) {
		pthread_join(threads[i], NULL);
	}

	queue_destroy(image_loaded_queue);
	queue_destroy(image_scaled_queue);
	queue_destroy(image_sharpenned_queue);
	queue_destroy(image_sobelled_queue);

	free(threads);
	printf("\n");
	return 0;
}
