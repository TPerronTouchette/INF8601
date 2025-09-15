#include <stdio.h>

extern "C" {
#include "filter.h"
#include "pipeline.h"
#include "image.h"
#include "tbb/pipeline.h"

}

class loadFilter : public tbb::filter_t<void, image_t*> {
    image_dir_t* const dir;

public:
    loadFilter(image_dir_t* d): dir(d) {} // llist initialization of constant private member dir

    image_t* operator()(tbb::flow_control& fc) {
        image_t* img = image_dir_load_next(dir);
        if (!img) {
            fc.stop();
            return nullptr;
        }
        return img;
    }
};

class scaleFilter : public tbb::filter_t<image_t*, image_t*> {

public:
    image_t* operator()(image_t* img, tbb::flow_control& fc) {
        if (!img) {
            fc.stop();
            return nullptr;
        }

        image_t* tempImg = filter_scale_up(img, 2);
        image_destroy(img); // destroys original image
        return tempImg;
    }
};

class sobelFilter : public tbb::filter_t<image_t*, image_t*> {
    public:
    image_t* operator()(image_t* img, tbb::flow_control& fc) {
        if (!img) {
            fc.stop(); 
            return nullptr;
        }

        image_t* tempImg = filter_sobel(img);
        image_destroy(img); // destroys original image
        return tempImg;
    }
};

class saveFilter : public tbb::filter_t<image_t*, void>{
    image_dir_t* const dir;

    public:
    saveFilter(image_dir_t* d): dir(d) {}

    void operator()(image_t* img, tbb::flow_control& fc) {
        if (!img) {
            fc.stop();
            return;
        }

        image_dir_save(dir, img);
        printf(".");
        fflush(stdout);
        image_destroy(img); // destroys original image
    }

};



int pipeline_tbb(image_dir_t* image_dir) {

    // tbb::filter_t myFilter = tbb::make_filter<Type1, Type2>(tbb::filter::mode, functor); where functor operator() maps Type1 to Type2
    tbb::filter_t load_filter = tbb::make_filter<void, image_t*>(tbb::filter::serial, loadFilter(image_dir));
    tbb::filter_t scale_filter = tbb::make_filter<image_t*, image_t*>(tbb::filter::parallel, scaleFilter());
    tbb::filter_t sobel_filter = tbb::make_filter<image_t*, image_t*>(tbb::filter::parallel, sobelFilter());
    tbb::filter_t save_filter = tbb::make_filter<image_t*, void>(tbb::filter::serial_out_of_order, saveFilter(image_dir));
    

    size_t max_tokens = 16;
    tbb::parallel_pipeline(max_tokens, load_filter & scale_filter & sobel_filter & save_filter);

        return 0;
}
