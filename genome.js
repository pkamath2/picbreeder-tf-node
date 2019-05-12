"use strict"

const tf = require('@tensorflow/tfjs-node');
const NodeGene = require('./nodegene');
const activationutil = require('./activationsutil');

class Genome {

    constructor(id, final_output_size, init_num_hidden_neurons) {
        this.id = id;
        this.final_output_size = final_output_size;
        this.init_num_hidden_neurons = init_num_hidden_neurons;

        this.global_innovation_number = 0; //Initialize global innovation number. In this version of CPPN, we will be creating Genome exactly once only. Subsequent only mutations occur. 
        this.node_gene_map = new Map(); //Innovation number -> Gene
    }

    incr_innovation_number() {
        return this.global_innovation_number = this.global_innovation_number + 1;
    }

    clone() {
        var new_genome = new Genome(this.id, this.final_output_size, this.init_num_hidden_neurons);
        new_genome.global_innovation_number = this.global_innovation_number;
        new_genome.node_gene_map = new Map();
        this.node_gene_map.forEach((node_gene, id) => {
            new_genome.node_gene_map.set(id, node_gene.clone());
        });
        return new_genome;
    }

    addNode(name, is_hidden, weight_seed, current_activation) {
        var activation = current_activation;
        if (activation == null) {
            activation = (name == 'output' ? activationutil.random_final_activation() : activationutil.random_activation());
            activation = ((name.indexOf('input') > -1) ? activationutil.random_input_activation() : activationutil.random_activation());
        }
        if (!weight_seed || weight_seed == null) weight_seed = Math.random();
        var nodeGene = new NodeGene((name == 'output' ? 100000 : this.incr_innovation_number()), name, is_hidden, activation, weight_seed, null, null, (name == 'output'), false);
        this.node_gene_map.set(nodeGene.innovation_number, nodeGene);
        return nodeGene;
    }

    addConnection(from_node_id, to_node_id) {
        var to_node = this.node_gene_map.get(to_node_id);
        to_node.from_conn_arr.push(from_node_id);
    }

    removeConnection(from_node_id, to_node_id) {
        var to_node = this.node_gene_map.get(to_node_id);
        var new_from_conn_arr = [];
        to_node.from_conn_arr.forEach(conn_id => {
            if (conn_id != from_node_id) new_from_conn_arr.push(conn_id);
        })
        to_node.from_conn_arr = new_from_conn_arr;
    }

    initializeGenome() {
        //Create input nodes
        for (var i = 0; i < 3; i++) { // 3 inputs - [t_x, t_y, t_r] 
            this.addNode('input_' + i, false, Math.random(), 'None');
        }
        //Create few hidden nodes
        for (var i = 0; i < 1; i++) {
            this.addNode('hidden_' + i, true);
        }
        //Create output node
        this.addNode('output', false);

        //Randomly connect nodes
        this.node_gene_map.forEach((node_gene, id) => {
            this.node_gene_map.forEach((node_gene_r, id_r) => {
                if (Math.random() > 0.7) {
                    if (!(node_gene.name.indexOf('input') > -1 && node_gene_r.name.indexOf('input') > -1)) {//Do not connect input to input
                        if (node_gene.innovation_number < node_gene_r.innovation_number) {//Always connect upwards. Hence output should have the highest innovation number. 
                            this.addConnection(node_gene.innovation_number, node_gene_r.innovation_number);
                        }
                    }
                }
            });
        });
        this.genomeHealthCheck();
    }

    initializeSingleGenome(latent_vector_flag) {
        var node_x = this.addNode('input_0', false, Math.random(), 'None');
        var node_y = this.addNode('input_1', false, Math.random(), 'None');
        var node_r = this.addNode('input_2', false, Math.random(), 'None');


        var hidden_1 = this.addNode('hidden_0', true, Math.random(), 'tf.tanh');
        var hidden_2 = this.addNode('hidden_1', true, Math.random(), 'tf.tanh');
        var hidden_3 = this.addNode('hidden_2', true, Math.random(), 'tf.tanh');
        var hidden_4 = this.addNode('hidden_3', true, Math.random(), 'tf.tanh');

        //Create output node
        var out = this.addNode('output', false, Math.random(), 'tf.sigmoid');

        //Fully connected network
        this.addConnection(node_x.innovation_number, hidden_1.innovation_number);
        this.addConnection(node_y.innovation_number, hidden_1.innovation_number);
        this.addConnection(node_r.innovation_number, hidden_1.innovation_number);

        if (latent_vector_flag) {
            var node_z = this.addNode('input_3', false, Math.random(), 'tf.tanh');
            this.addConnection(node_z.innovation_number, hidden_1.innovation_number);
        }

        this.addConnection(hidden_1.innovation_number, hidden_2.innovation_number);
        this.addConnection(hidden_2.innovation_number, hidden_3.innovation_number);
        this.addConnection(hidden_3.innovation_number, hidden_4.innovation_number);
        this.addConnection(hidden_4.innovation_number, out.innovation_number);
        // this.addConnection(hidden_1.innovation_number, hidden_4.innovation_number);
        // this.addConnection(hidden_4.innovation_number, out.innovation_number);

        this.genomeHealthCheck();
    }

    genomeHealthCheck() {//Disables nodes which do not have any inputs. 

        var max_innovation_number = 0;
        this.node_gene_map.forEach((node_gene, id) => {
            //1. If the output node has no incoming connections, atleast feed it one input.
            if (node_gene.is_output_node && node_gene.from_conn_arr.length == 0) {
                this.node_gene_map.forEach((node_gene_r) => {
                    if (node_gene_r.name.indexOf('input') > -1 && !node_gene_r.disabled) {
                        node_gene.from_conn_arr.push(node_gene_r.innovation_number);
                    }
                });
            }

            //2. //If there is no input, then disable.Except for input nodes.
            if (!(node_gene.name.indexOf('input') > -1) && (node_gene.from_conn_arr.length == 0)) {
                node_gene.disabled = true;
            }

            //3. Correctly set the max global innovation number
            if (id != 100000 && max_innovation_number < id) max_innovation_number = id;

        });
        this.global_innovation_number = max_innovation_number;

        this.sortByEvaluationSequence();
    }

    static reconstructGenome(genome, final_output_size, num_hidden_neurons) {
        var new_genome = new Genome(genome.genome_id, final_output_size, num_hidden_neurons);
        var node_gene_map = new Map();
        var gene_arr = genome.node_genes;
        var max_innovation_number = 0;
        gene_arr.forEach(node_gene => {
            node_gene.innovation_number = +node_gene.innovation_number;
            node_gene.hidden = node_gene.hidden === 'true';//Converting String to boolean
            node_gene.is_output_node = node_gene.is_output_node === 'true';
            node_gene.disabled = node_gene.disabled === 'true';
            var new_conn_arr = [];
            if (node_gene.from_conn_arr) {
                node_gene.from_conn_arr.forEach(c => {
                    new_conn_arr.push(+c);
                })
            }
            node_gene.from_conn_arr = new_conn_arr;
            if (node_gene.innovation_number != 100000 && max_innovation_number < node_gene.innovation_number) max_innovation_number = node_gene.innovation_number;
            var new_node_gene = new NodeGene(node_gene.innovation_number, node_gene.name, node_gene.hidden, node_gene.activation, node_gene.weight_seed, node_gene.from_conn_arr, node_gene.to_conn_arr, node_gene.is_output_node, node_gene.disabled);
            node_gene_map.set(new_node_gene.innovation_number, new_node_gene);
        });
        new_genome.global_innovation_number = max_innovation_number + 1;
        var node_gene_ids = [...node_gene_map.keys()];
        node_gene_ids = node_gene_ids.sort((a, b) => a - b);
        var sorted_node_gene_map = new Map();
        for (var i = 0; i < node_gene_ids.length; i++) {
            sorted_node_gene_map.set(node_gene_ids[i], node_gene_map.get(node_gene_ids[i]));
        }
        new_genome.node_gene_map = sorted_node_gene_map;
        return new_genome;
    }

    evaluate(inputs) {
        return tf.tidy(() => {
            var outputs_map = new Map();
            this.node_gene_map.forEach((node_gene, id) => {
                var input_shape = inputs[0].shape;
                var node_output_size = (node_gene.is_output_node) ? this.final_output_size : this.init_num_hidden_neurons;
                if (!node_gene.disabled) {
                    try {
                        var input = null;
                        if (node_gene.name.indexOf('input') > -1) {
                            input = inputs[node_gene.name.substring('input_'.length)];
                            var with_bias = node_gene.name.substring('input_'.length) == 3 ? true : false;
                            outputs_map.set(id, node_gene.evaluate(input, node_output_size, with_bias, true));//shape doesnt matter. We only need activation here.
                        } else {
                            var from_node_ids = node_gene.from_conn_arr;
                            var connInput = null;//tricky
                            for (var j = 0; j < from_node_ids.length; j++) {
                                if (connInput == null) connInput = tf.zeros([input_shape[0], this.init_num_hidden_neurons]);
                                connInput = tf.add(connInput, outputs_map.get(from_node_ids[j]));
                            }
                            outputs_map.set(id, node_gene.evaluate(connInput, node_output_size, true, true));
                        }
                    } catch (error) {
                        console.log(node_gene)
                        throw error;
                    }
                } else {
                    outputs_map.set(id, tf.zeros([input_shape[0], this.init_num_hidden_neurons]));
                }
            });
            return { output: outputs_map.get(100000), node_genes: [...this.node_gene_map.values()] };
        });

    }

    sortByEvaluationSequence() {
        this.node_gene_map.forEach((node_gene, id) => {
            // console.log('Calculating complexity for -- ')
            // console.log(node_gene)
            node_gene.computational_complexity = this.findComputationalComplexity(node_gene);
        });

        var sorted_nodes = [...this.node_gene_map.values()];
        sorted_nodes = sorted_nodes.sort((a, b) => a.computational_complexity - b.computational_complexity);
        var sorted_nodes_map = new Map();
        sorted_nodes.forEach((node_gene) => {
            sorted_nodes_map.set(node_gene.innovation_number, node_gene);
        })
        this.node_gene_map = sorted_nodes_map;
    }

    findComputationalComplexity(node_gene) {
        var complexity = 0;
        // console.log("Computing complexity for: " + node_gene.innovation_number + ":" + node_gene.activation);
        node_gene.from_conn_arr.forEach(c => {
            complexity++;
            var sub_complexity = this.findComputationalComplexity(this.node_gene_map.get(c));
            complexity = complexity + sub_complexity;
        });
        return complexity;
    }
}
module.exports = Genome;