const tf = require('@tensorflow/tfjs-node');
const Genome = require('./genome');

class Mutator{

    cross(genome_1, genome_2){
        return tf.tidy(() => {
            var crossed_gene_map = new Map();
            genome_1.node_gene_map.forEach((node_gene_1, id_1) => {
                genome_2.node_gene_map.forEach((node_gene_2, id_2) => {
                    if(id_1 == id_2){
                        if(Math.random() > 0.5){
                            crossed_gene_map.set(id_1, node_gene_1); //Common set
                        }else{
                            crossed_gene_map.set(id_2, node_gene_2); //Common set
                        }
                    }
                });
            });
    
            var crossed_ids = [...crossed_gene_map.keys()];
            genome_1.node_gene_map.forEach((node_gene_1, id_1) => {
                if(crossed_ids.indexOf(id_1)<0){
                    crossed_gene_map.set(id_1, node_gene_1);//Disjoint set from set 1.
                }
            });
            genome_2.node_gene_map.forEach((node_gene_2, id_2) => {
                if(crossed_ids.indexOf(id_2)<0){
                    crossed_gene_map.set(id_2, node_gene_2);//Disjoint set from set 2.
                }
            });
    
            var crossedGenome = new Genome(Math.random()*10, genome_1.final_output_size, genome_1.init_num_hidden_neurons);
    
            crossedGenome.node_gene_map = crossed_gene_map;
            crossedGenome.genomeHealthCheck();
            return crossedGenome;
        });
    }

    mutate(genome){
        return tf.tidy(() => {
            var node_gene_map = genome.node_gene_map;
            var connection_arr = [];
            var hidden_count = 0;
            node_gene_map.forEach((node_gene, id) => {
                var from_conn_arr = node_gene.from_conn_arr;
                from_conn_arr.forEach(id_1 => {
                    connection_arr.push(id_1+"_"+id); //Just a simple representation indicating from node -> to node connection. 
                });
                if(node_gene.is_hidden) hidden_count++;
            });
    
            var new_node_position = Math.floor(Math.random()*connection_arr.length);
            var curr_conn = connection_arr[new_node_position];
            var node_from_id = +curr_conn.split("_")[0];
            var node_to_id = +curr_conn.split("_")[1];
            genome.removeConnection(node_from_id, node_to_id);
    
            var node_new = genome.addNode('hidden_'+(hidden_count+1), true, 1);//weight_seed = 1 for new node.
            genome.addConnection(node_from_id, node_new.innovation_number);
            genome.addConnection(node_new.innovation_number, node_to_id);
    
            var node_ids = [...genome.node_gene_map.keys()];
            node_ids = node_ids.sort((a,b)=>a-b);
            var sorted_node_gene_map = new Map();
            for(var i=0;i<node_ids.length;i++){
                sorted_node_gene_map.set(node_ids[i], genome.node_gene_map.get(node_ids[i]));
            }
            genome.node_gene_map = sorted_node_gene_map;
            genome.genomeHealthCheck();
    
            return genome;
        });
    }

}
module.exports = Mutator;