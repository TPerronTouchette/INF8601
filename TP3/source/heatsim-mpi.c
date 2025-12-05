#include <assert.h>
#include <stddef.h>

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
    MPI_Cart_create(MPI_COMM_WORLD,2,dims,periods,0,&heatsim->communicator);

    // Error Handling

    // Setting other parameters in heatsim
    MPI_Comm_size(heatsim->communicator, &heatsim->rank_count);
    MPI_Comm_rank(heatsim->communicator, &heatsim->rank);
    
    MPI_Cart_shift(heatsim->communicator,0,1,&heatsim->rank_south_peer,&heatsim->rank_north_peer);
    MPI_Cart_shift(heatsim->communicator,1,1,&heatsim->rank_west_peer, &heatsim->rank_east_peer);

    MPI_Cart_coords(heatsim->communicator,heatsim->rank,2,heatsim->coordinates);

    


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
    int size;
    int coords[2];
    unsigned int buffer[3]; 
    grid_t* grid ;
    MPI_Comm_size(heatsim->communicator, &size);

    MPI_Datatype grid_data_t;

    // For MPI_Struct
    int count;
    int array_of_block_lengths[1]; // Number of elements in the grid
    MPI_Aint array_of_displacements[1];
    int array_of_types[1];
 
    for (int rank = 1 ; rank < size ; ++rank){

        MPI_Cart_coords(heatsim->communicator,rank,2,coords);
        grid = cart2d_get_grid(cart,coords[0],coords[1]);

        buffer[0] = grid->width;
        buffer[1] = grid->height;
        buffer[2] = grid->padding;

        MPI_Send(buffer,3,MPI_UNSIGNED,rank,0,heatsim->communicator);


        count = 1;
        array_of_block_lengths[0] = grid->width * grid->height;
        array_of_displacements[0] = 0;
        array_of_types[0] = MPI_DOUBLE;

        MPI_Type_create_struct(count, array_of_block_lengths, array_of_displacements, array_of_types, &grid_data_t);
        MPI_Type_commit(&grid_data_t);

        MPI_Send(grid->data,1,grid_data_t,rank,0,heatsim->communicator);

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
    MPI_Status status;
    MPI_Recv(buffer,3,MPI_UNSIGNED,0,0,heatsim->communicator,&status);
    grid_t* grid = grid_create(buffer[0],buffer[1], buffer[2]);

    int count = 1;
    int array_of_block_lengths[1] = {grid->width * grid->height};
    MPI_Aint array_of_displacements[1] = {0};
    int array_of_types[1] = {MPI_DOUBLE};
    MPI_Datatype grid_data_t;

    MPI_Type_create_struct(count, array_of_block_lengths, array_of_displacements, array_of_types, &grid_data_t);
    MPI_Type_commit(&grid_data_t);
    MPI_Recv(grid->data,1,grid_data_t,0,0,heatsim->communicator, &status);
    MPI_Type_free(&grid_data_t);

    return 0;

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

fail_exit:
    return -1;
}

int heatsim_send_result(heatsim_t* heatsim, grid_t* grid) {
    assert(grid->padding == 0);

    /*
     * TODO: Envoyer les données (`data`) du `grid` résultant au rang 0. Le
     *       `grid` n'a aucun rembourage (padding = 0);
     */

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

fail_exit:
    return -1;
}
