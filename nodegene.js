"use strict"

const tf = require('@tensorflow/tfjs-node');
const activationutil = require('./activationsutil');

class NodeGene{

    constructor(innovation_number, name, hidden, activation, weight_seed, from_conn_arr, to_conn_arr, is_output_node, disabled){
        this.innovation_number = innovation_number;
        this.name = name;
        this.hidden = hidden;
        this.activation = activation;
        this.weight_seed = weight_seed;
        if(!from_conn_arr) from_conn_arr = [];
        this.from_conn_arr = from_conn_arr;
        // if(!to_conn_arr) to_conn_arr = [];
        // this.to_conn_arr = to_conn_arr;//Do not maintain this anymore.
        this.is_output_node = is_output_node;
        this.disabled = disabled;
        this.computational_complexity = 0;
    }

    incr_computational_complexity(){
        this.computational_complexity++;
    }

    clone(){
        var new_node_gene = new NodeGene(this.innovation_number, this.name, this.hidden, this.activation, this.weight_seed, this.from_conn_arr, null, this.is_output_node, this.disabled);
        new_node_gene.computational_complexity = this.computational_complexity;

        return new_node_gene;
    }

    evaluate(input, output_size, with_bias, with_weights){
        if(this.disabled) {
            return tf.zeros(input.shape);
        }

        var result = input;
        if(output_size==null) output_size= 64;
        if(with_bias ==null) with_bias = false;
        
        if(with_weights){//inputs dont need matmul in the first round.
            var weights;
            if(this.weight_seed == 1){// For future - Mutation. Add Connection, new incoming conn has weight of 1.
                weights = tf.variable(tf.ones([input.shape[1], output_size]));
            }else{
                weights = tf.variable(tf.randomNormal([input.shape[1], output_size],0.0,1.0,'float32', this.weight_seed));
            }
            result = tf.matMul(input, weights);
            if(with_bias){
                var bias = tf.variable(tf.randomNormal([1, output_size],0.0,1.0,'float32', this.weight_seed));
                result = tf.add(result, tf.mul(bias, tf.ones([input.shape[0],1])));
            }
        }
        if(this.name.indexOf('input')>-1 && this.activation=='tf.sin') {result = tf.mul(result, 10);}
        return activationutil.enum_activations[this.activation](result);
    }

}
module.exports = NodeGene;