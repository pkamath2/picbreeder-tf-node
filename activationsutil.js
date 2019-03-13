"use strict"

const tf = require('@tensorflow/tfjs-node');

class activationutil{

    static get enum_activations() {
        return {
            'tf.tanh':tf.tanh,
            'tf.softplus':tf.softplus,
            'tf.sin': tf.sin,
            'activationutil.gaussian':activationutil.gaussian,
            'activationutil.tanh':activationutil.tanh,
            'tf.sigmoid':tf.sigmoid,
            'activationutil.psychedelic':activationutil.psychedelic
        };
    }

    static gaussian(d){ //Assuming mean=0.0 and stddev=1.0
        var g_std_dev = 1.0;
        var g_mean = 0.0;
        return tf.div(tf.pow(Math.E, tf.div(tf.pow(tf.div(tf.sub(d, g_mean), g_std_dev),2),-2.0)),Math.sqrt(g_std_dev*2*Math.PI));
    }

    static psychedelic(d){//Found this gem in David Ha's python notebook at - https://github.com/hardmaru/cppn-tensorflow
        return tf.add(tf.mul(tf.sin(d),0.5),0.5);
    }

    static tanh(d){
        return tf.abs(tf.mul(tf.tanh(d),0.001));//Probable scaling bug by using 0.9! =(
    }

    static activations(ind){
        return ['tf.tanh','tf.softplus', 'tf.sin','activationutil.gaussian', 'tf.sigmoid'][ind];
    }

    static final_activations(ind){//Reduce the probablity of getting psychedelic activation.
        return ['activationutil.gaussian', 'activationutil.tanh', 'tf.sigmoid', 'activationutil.psychedelic'][ind];
    }

    static random_activation(){
        return activationutil.activations(Math.floor(Math.random()*(5)));
    }

    static random_final_activation(){
        return activationutil.final_activations(Math.floor(Math.random()*(4)));
    }

    static random_input_activation(){
        if(Math.random()>0.6) return 'tf.sin';
        else return 'tf.tanh';
    }
}
module.exports = activationutil;