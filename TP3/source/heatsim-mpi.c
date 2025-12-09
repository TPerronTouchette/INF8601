#include <assert.h>
#include <stddef.h>
#include <stdbool.h>

#include "heatsim.h"
#include "log.h"


int heatsim_init(heatsim_t* heatsim, unsigned int dim_x, unsigned int dim_y) {
    /*
     * TODO: Initialiser tous les membres de la structure `heatsim`.
     *       Le communicateur doit être périodique. Le communicateur
     *       cartésien est périodique en X et Y.
     */
    
    // Create a periodical cartesian communicator
    const int dims[2] = {dim_x,dim_y};
    const int periods[2] = {1, 1};


    int ret = MPI_SUCCESS;
    ret = MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,0,&heatsim->communicator);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("MPI_Cart_create failed", ret);
        goto fail_exit;
    }

    // Error Handling

    // Setting other parameters in heatsim
    ret = MPI_Comm_size(heatsim->communicator, &heatsim->rank_count);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("MPI_Comm_size failed", ret);
        goto fail_exit;
    }

    ret = MPI_Comm_rank(heatsim->communicator, &heatsim->rank);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("MPI_Comm_rank failed", ret);
        goto fail_exit;
    }    

    ret = MPI_Cart_shift(heatsim->communicator,0,1,&heatsim->rank_west_peer,&heatsim->rank_east_peer);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("MPI_Cart_shift failed", ret);
        goto fail_exit;
    } 

    ret = MPI_Cart_shift(heatsim->communicator,1,1,&heatsim->rank_north_peer, &heatsim->rank_south_peer);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("MPI_Cart_shift failed", ret);
        goto fail_exit;
    }

    ret = MPI_Cart_coords(heatsim->communicator,heatsim->rank,2,heatsim->coordinates);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("MPI_Cart_coords failed", ret);
        goto fail_exit;
    }

    


    return 0;

fail_exit:
    return -1;
}

int heatsim_send_grids(heatsim_t* heatsim, cart2d_t* cart) {
    /*
     * TODO: Envoyer toutes les `grid` aux autres rangs. Cette fonction
     *       est appelé pour le rang 0. Par exemple, si le rang 3 est à la
     *       coordonnée cartésienne (0, 2), alors on y envoit le `grid`
     *       à la position (0, 2) dans `cart`.
     *
     *       Il est recommandé d'envoyer les paramètres `width`, `height`
     *       et `padding` avant les données. De cette manière, le receveur
     *       peut allouer la structure avec `grid_create` directement.
     *
     *       Utilisez `cart2d_get_grid` pour obtenir la `grid` à une coordonnée.
     */
    int coords[2];
    unsigned int buffer[3]; 
    grid_t* grid ;
    int ret = MPI_SUCCESS;
    MPI_Datatype grid_data_t;

    // For MPI_Struct
    int count;
    int array_of_block_lengths[1]; // Number of elements in the grid
    MPI_Aint array_of_displacements[1];
    int array_of_types[1];
 
    for (int rank = 1 ; rank < heatsim->rank_count ; ++rank){

        MPI_Cart_coords(heatsim->communicator,rank,2,coords);
        grid = cart2d_get_grid(cart,coords[0],coords[1]);

        buffer[0] = grid->width;
        buffer[1] = grid->height;
        buffer[2] = grid->padding;

        ret = MPI_Send(buffer,3,MPI_UNSIGNED,rank,0,heatsim->communicator);


        count = 1;
        array_of_block_lengths[0] = grid->width * grid->height;
        array_of_displacements[0] = 0;
        array_of_types[0] = MPI_DOUBLE;

        MPI_Type_create_struct(count, array_of_block_lengths, array_of_displacements, array_of_types, &grid_data_t);
        MPI_Type_commit(&grid_data_t);

        ret = MPI_Send(grid->data,1,grid_data_t,rank,0,heatsim->communicator);

        MPI_Type_free(&grid_data_t);
        

    }
    return 0;


fail_exit:
    return -1;
}

grid_t* heatsim_receive_grid(heatsim_t* heatsim) {
    /*
     * TODO: Recevoir un `grid ` du rang 0. Il est important de noter que
     *       toutes les `grid` ne sont pas nécessairement de la même
     *       dimension (habituellement ±1 en largeur et hauteur). Utilisez
     *       la fonction `grid_create` pour allouer un `grid`.
     *
     *       Utilisez `grid_create` pour allouer le `grid` à retourner.
     */
    unsigned int buffer[3];
    int ret = MPI_SUCCESS;

    ret = MPI_Recv(buffer,3,MPI_UNSIGNED,0,0,heatsim->communicator,MPI_STATUS_IGNORE);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("MPI_Recv grid failed: ", ret);
        goto fail_exit;
    }

    grid_t* grid = grid_create(buffer[0],buffer[1], buffer[2]);
    if (!grid) goto fail_exit;

    int count = 1;
    int array_of_block_lengths[1] = {grid->width * grid->height};
    MPI_Aint array_of_displacements[1] = {0};
    int array_of_types[1] = {MPI_DOUBLE};
    MPI_Datatype grid_data_t;


    ret = MPI_Type_create_struct(count, array_of_block_lengths, array_of_displacements, array_of_types, &grid_data_t);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("MPI_Type_create_struct failed", ret);
        goto fail_exit;
    }

    ret = MPI_Type_commit(&grid_data_t);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("MPI_Type_commit failed", ret);
        goto fail_exit;
    }

    ret = MPI_Recv(grid->data,1,grid_data_t,0,0,heatsim->communicator, MPI_STATUS_IGNORE);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("MPI_Cart_create failed", ret);
        goto fail_exit;
    }

    ret = MPI_Type_free(&grid_data_t);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("MPI_Type_free failed", ret);
        goto fail_exit;
    }

    return grid;

fail_exit:
    return NULL;
}

int heatsim_exchange_borders(heatsim_t* heatsim, grid_t* grid) {
    assert(grid->padding == 1);

    /*
     * TODO: Échange les bordures de `grid`, excluant le rembourrage, dans le
     *       rembourrage du voisin de ce rang. Par exemple, soit la `grid`
     *       4x4 suivante,
     *
     *                            +-------------+
     *                            | x x x x x x |
     *                            | x A B C D x |
     *                            | x E F G H x |
     *                            | x I J K L x |
     *                            | x M N O P x |
     *                            | x x x x x x |
     *                            +-------------+
     *
     *       où `x` est le rembourrage (padding = 1). Ce rang devrait envoyer
     *
     *        - la bordure [A B C D] au rang nord,
     *        - la bordure [M N O P] au rang sud,
     *        - la bordure [A E I M] au rang ouest et
     *        - la bordure [D H L P] au rang est.
     *
     *       Ce rang devrait aussi recevoir dans son rembourrage
     *
     *        - la bordure [A B C D] du rang sud,
     *        - la bordure [M N O P] du rang nord,
     *        - la bordure [A E I M] du rang est et
     *        - la bordure [D H L P] du rang ouest.
     *
     *       Après l'échange, le `grid` devrait avoir ces données dans son
     *       rembourrage provenant des voisins:
     *
     *                            +-------------+
     *                            | x m n o p x |
     *                            | d A B C D a |
     *                            | h E F G H e |
     *                            | l I J K L i |
     *                            | p M N O P m |
     *                            | x a b c d x |
     *                            +-------------+
     *
     *       Utilisez `grid_get_cell` pour obtenir un pointeur vers une cellule.
     */
    
    int north = heatsim->rank_north_peer;
    int south = heatsim->rank_south_peer;
    int east = heatsim->rank_east_peer;
    int west = heatsim->rank_west_peer;
    int rank = heatsim->rank;

    printf("[%d] Grid width : %d, height: %d, width_padded: %d, height_padded: %d\n", heatsim->rank,grid->width, grid->height, grid->width_padded, grid->height_padded);


    MPI_Datatype column_t;
    int ret = MPI_SUCCESS;

    ret = MPI_Type_vector(grid->height,1,grid->width_padded,MPI_DOUBLE, &column_t);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("MPI_Type_vector failed", ret);
        goto fail_exit;
    }

    ret = MPI_Type_commit(&column_t);
    if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("MPI_Type_commit failed", ret);
        goto fail_exit;
    }


    
    // NORTH AND SOUTH NEIGHBORS :


    // handling if no neighbors (one process)
    if(north == rank && south == rank){

        // north to south
        memcpy(grid_get_cell(grid,0,grid->height), grid_get_cell(grid,0,0), grid->width * sizeof(double));

        // south to north
        memcpy(grid_get_cell(grid,0,-1), grid_get_cell(grid,0,grid->height-1), grid->width * sizeof(double));
    }
    else {
        // Retreiving border cells
        double* buffer_north = malloc(grid->width * sizeof(double));
        double* buffer_south = malloc(grid->width * sizeof(double));

        // Interior boundries
        for (int i = 0; i < grid->width; ++i){
            buffer_north[i] = *grid_get_cell(grid,i,0);
            buffer_south[i] = *grid_get_cell(grid,i, grid->height-1);
        }

        // Send To North
        printf("[%d] sending to north\n",heatsim->rank);
        ret = MPI_Send(buffer_north,grid->width,MPI_DOUBLE,north,0,heatsim->communicator);
        if (ret != MPI_SUCCESS){
            LOG_ERROR_MPI("MPI_Send North failed: ", ret);
            goto fail_exit;
        }
        printf("[%d] sent to north [%d]\n", heatsim->rank,heatsim->rank_north_peer);
        free(buffer_north);

        // Send To South
        printf("[%d] sending to south\n",heatsim->rank);
        ret = MPI_Send(buffer_south,grid->width,MPI_DOUBLE,south,1,heatsim->communicator);
        if (ret != MPI_SUCCESS){
            LOG_ERROR_MPI("MPI_Send South failed: ", ret);
            goto fail_exit;
        }
        printf("[%d] sent to south [%d]\n",heatsim->rank, heatsim->rank_south_peer);
        free(buffer_south);

        // Receive From South
        printf("[%d] receiving from South\n",heatsim->rank);
        ret = MPI_Recv(grid_get_cell(grid,0,grid->height),grid->width,MPI_DOUBLE,south,0,heatsim->communicator,MPI_STATUS_IGNORE);
        if (ret != MPI_SUCCESS){
            LOG_ERROR_MPI("MPI_Recv South failed: ", ret);
            goto fail_exit;
        }
        printf("[%d] received from South [%d]\n",heatsim->rank, heatsim->rank_south_peer);
        // Receive From North
        printf("[%d] receiving from North\n",heatsim->rank);
        ret = MPI_Recv(grid_get_cell(grid,0,-1),grid->width,MPI_DOUBLE,north,1,heatsim->communicator,MPI_STATUS_IGNORE);
        if (ret != MPI_SUCCESS){
            LOG_ERROR_MPI("MPI_Recv North failed: ", ret);
            goto fail_exit;
        }
        printf("[%d] received from North [%d]\n",heatsim->rank, heatsim->rank_north_peer);



    }


    // EAST AND WEST NEIGHBORS :

    // handling if no neighbors (one process)
    if(west == rank && east == rank){
        for (int i = 0; i < grid->height; ++i){
            // west to east
            memcpy(grid_get_cell(grid,grid->width,i), grid_get_cell(grid,0,i), sizeof(double));
            // east to west
            memcpy(grid_get_cell(grid,-1,i), grid_get_cell(grid,grid->width-1,i), sizeof(double));
        }
    }

    else {
        // Send To West
        ret = MPI_Send(grid_get_cell(grid,0,0),1,column_t,heatsim->rank_west_peer,2,heatsim->communicator);
        if (ret != MPI_SUCCESS){
            LOG_ERROR_MPI("MPI_Send West failed: ", ret);
            goto fail_exit;
        }
        // Send To East
        ret = MPI_Send(grid_get_cell(grid,grid->width-1,0),1,column_t,heatsim->rank_east_peer,3,heatsim->communicator);
        if (ret != MPI_SUCCESS){
            LOG_ERROR_MPI("MPI_Send East failed: ", ret);
            goto fail_exit;
        }
        // Receive From East
        ret = MPI_Recv(grid_get_cell(grid,grid->width,0),grid->height,column_t,heatsim->rank_east_peer,2,heatsim->communicator,MPI_STATUS_IGNORE);
        if (ret != MPI_SUCCESS){
            LOG_ERROR_MPI("MPI_Recv East failed: ", ret);
            goto fail_exit;
        }
        // Receive From West
        ret = MPI_Recv(grid_get_cell(grid,-1,0),grid->height,column_t,heatsim->rank_west_peer,3,heatsim->communicator,MPI_STATUS_IGNORE);
        if (ret != MPI_SUCCESS){
            LOG_ERROR_MPI("MPI_Recv West failed: ", ret);
            goto fail_exit;
        }


    }

    ret = MPI_Type_free(&column_t);
    if (ret != MPI_SUCCESS){
        LOG_ERROR_MPI("MPI_Type_free failed", ret);
        goto fail_exit; 
    }

    return 0;

fail_exit:
    return -1;
}


int heatsim_send_result(heatsim_t* heatsim, grid_t* grid) {
    assert(grid->padding == 0);

    /*
     * TODO: Envoyer les données (`data`) du `grid` résultant au rang 0. Le
     *       `grid` n'a aucun rembourage (padding = 0);
     */
    MPI_Request req;
    int ret = MPI_SUCCESS;
    ret = MPI_Isend(grid->data,grid->width * grid->height, MPI_DOUBLE,0,0,heatsim->communicator,&req);
    if (ret != MPI_SUCCESS){
        LOG_ERROR_MPI("MPI_Isend failed", ret);
        goto fail_exit; 
    }

    ret = MPI_Wait(&req,MPI_STATUS_IGNORE);
    if (ret != MPI_SUCCESS){
        LOG_ERROR_MPI("MPI_Wait failed", ret);
        goto fail_exit; 
    }
    return 0;

fail_exit:
    return -1;
}

int heatsim_receive_results(heatsim_t* heatsim, cart2d_t* cart) {
    /*
     * TODO: Recevoir toutes les `grid` des autres rangs. Aucune `grid`
     *       n'a de rembourage (padding = 0).
     *
     *       Utilisez `cart2d_get_grid` pour obtenir la `grid` à une coordonnée
     *       qui va recevoir le contenue (`data`) d'un autre noeud.
     */

    int coords[2];
    grid_t* grid;
    MPI_Request *reqs = malloc((heatsim->rank_count -1) * sizeof(MPI_Request));
    int req_id = 0;
    int ret = MPI_SUCCESS;

    for (int rank = 1 ; rank < heatsim->rank_count ; ++rank){
        ret = MPI_Cart_coords(heatsim->communicator,rank,2,coords);
        if (ret != MPI_SUCCESS){
            LOG_ERROR_MPI("MPI_Cart_coords failed", ret);
            goto fail_exit; 
        }
        grid = cart2d_get_grid(cart,coords[0],coords[1]);
        ret = MPI_Irecv(grid->data,grid->width * grid->height,MPI_DOUBLE,rank,0,heatsim->communicator,&reqs[req_id]);
        if (ret != MPI_SUCCESS){
            LOG_ERROR_MPI("MPI_Irecv failed", ret);
            goto fail_exit; 
        }
        req_id++;
    }

    ret = MPI_Waitall(heatsim->rank_count - 1 , reqs, MPI_STATUSES_IGNORE);
    if (ret != MPI_SUCCESS){
        LOG_ERROR_MPI("MPI_Waitall failed", ret);
        goto fail_exit; 
    }

    free(reqs);
    return 0;

fail_exit:
    return -1;
}

