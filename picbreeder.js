"use strict"

const tf = require('@tensorflow/tfjs-node')
const express = require('express');
const bodyParser = require('body-parser');
const {createCanvas, createImageData} = require('canvas');
const Genome = require('./genome');
const Mutator = require('./mutator');

/**Express props */
const port = 9995;
/*Thumbnail props*/
const height = 120;
const width = 136;
/*Genome Props*/
let num_hidden_neurons = 8;
let num_output = 1;
/*Montage props*/
const montage_size = 25; //Always use a proper square

function createInputs(width, height){
    let t_x = tf.div(tf.div(tf.mul(tf.sub(tf.range(0, width), (width - 1) / 2.0), 1.0), (width - 1)), 0.5);
    t_x = t_x.reshape([1, width]);
    t_x = tf.matMul(tf.ones([height, 1], 'float32'), t_x);
    t_x = t_x.flatten().reshape([height * width, 1])
    
    let t_y = tf.div(tf.div(tf.mul(tf.sub(tf.range(0, height), (height - 1) / 2.0), 1.0), (height - 1)), 0.5);
    t_y = t_y.reshape([height, 1]);
    t_y = tf.matMul(t_y, tf.ones([1, width], 'float32'));
    t_y = t_y.flatten().reshape([height * width, 1])
    
    /*Initialize r*/
    let t_r = tf.sqrt(tf.add(tf.pow(t_x, 2), tf.pow(t_y, 2)));
    t_r = tf.transpose(t_r).flatten().reshape([height * width, 1])
    
    return [t_x, t_y, t_r];
}

function main() {
    const app = express();
    app.listen(process.env.PORT || port, () => console.log(`Example app listening on port ${port}!`));
    app.use(bodyParser.urlencoded({extended:true}));
    app.use(bodyParser.json());
    
    app.get('/', function(req, res){
        res.sendFile(__dirname+'/views/index.html');
    });

    app.get('/montage', function (req, res) {
        if(req.query.color == 'true') {
            num_output = 3;
        }else{
            num_output = 1;
        }
        if(req.query.dense =='true') { 
            num_hidden_neurons = 64; 
        }else{
            num_hidden_neurons = 32; 
        }
        res.on('finish', () => { console.log('Response sent.') });
        let dataArr = [];
        for(var m_ind=0;m_ind<montage_size;m_ind++){
            var inputs = createInputs(width, height);

            var genome = new Genome(m_ind, num_output, num_hidden_neurons);
            genome.initializeGenome(); 
            
            var outputs = genome.evaluate([inputs[0], inputs[1], inputs[2]]);
            var node_genes = outputs.node_genes;
            var data = outputs.output;
            if(num_output==3) data = data.reshape([data.shape[0]*3])//RGB Channels
            
            //Duplicated from Tensorflow array_ops.toPixels(). That method does not work with await.
            data = data.dataSync();
            var bytes = new Array(width * height * 4); //Convert to Uint8 array on the UI
            var r,g,b,a,j,i;
            for (i = 0; i < height * width; ++i) {
                r = void 0, g = void 0, b = void 0, a = void 0;
                if(num_output == 3){
                    r = data[i * 3] * 255;
                    g = data[i * 3 + 1] * 255;
                    b = data[i * 3 + 2] * 255;
                }else{
                    r = data[i] * 255;
                    g = data[i] * 255;
                    b = data[i] * 255;
                }
                a = 255;
                j = i * 4;
                bytes[j + 0] = Math.round(r);
                bytes[j + 1] = Math.round(g);
                bytes[j + 2] = Math.round(b);
                bytes[j + 3] = Math.round(a);
            }
            dataArr.push({data:bytes, genome_id: genome.id, node_genes:node_genes});
        }
        res.send(dataArr);
    });

    app.post('/mutate', function(req, res){
        let params = req.body;
        if(params.color == 'true') {
            num_output = 3;
        }else{
            num_output = 1;
        }
        if(params.dense =='true') { 
            num_hidden_neurons = 64; 
        }else{
            num_hidden_neurons = 32; 
        }

        var selected_genomes = [];
        params.genomes.forEach(genomeJson => {
            var genome = Genome.reconstructGenome(genomeJson, num_output, num_hidden_neurons);
            selected_genomes.push(genome);
        });
        res.on('finish', () => { console.log('Response sent.') });
        let dataArr = [];

        var num_mutation_1 = Math.floor(0.25*montage_size);
        var num_mutation_2 = Math.floor(0.25*montage_size);
        var num_cross = Math.ceil(0.5*montage_size);

        for(var m_ind=0;m_ind<num_mutation_1;m_ind++){
            var inputs = createInputs(width, height);
            var mutator = new Mutator();
            var mutated_genome = mutator.mutate(selected_genomes[0].clone())
            var outputs = mutated_genome.evaluate([inputs[0], inputs[1], inputs[2]]);
            var node_genes = outputs.node_genes;
            var data = outputs.output;
            
            if(num_output==3) data = data.reshape([data.shape[0]*3])//RGB Channels
            var bytes = convertToByteArray(data.dataSync(), width*height);
            dataArr.push({data:bytes, genome_id: mutated_genome.id, node_genes:node_genes});
        }

        for(var m_ind=0;m_ind<num_mutation_2;m_ind++){
            var inputs = createInputs(width, height);
            var mutator = new Mutator();
            var mutated_genome = mutator.mutate(selected_genomes[1].clone())
            var outputs = mutated_genome.evaluate([inputs[0], inputs[1], inputs[2]]);
            var node_genes = outputs.node_genes;
            var data = outputs.output;
            if(num_output==3) data = data.reshape([data.shape[0]*3])//RGB Channels
            var bytes = convertToByteArray(data.dataSync(), width*height);
            dataArr.push({data:bytes, genome_id: mutated_genome.id, node_genes:node_genes});
        }

        for(var m_ind=0;m_ind<num_cross;m_ind++){
            var inputs = createInputs(width, height);
            var mutator = new Mutator();
            var crossed_genome = mutator.cross(selected_genomes[0], selected_genomes[1]);
            var outputs = crossed_genome.evaluate([inputs[0], inputs[1], inputs[2]]);
            var node_genes = outputs.node_genes;
            var data = outputs.output;
            if(num_output==3) data = data.reshape([data.shape[0]*3])//RGB Channels
            var bytes = convertToByteArray(data.dataSync(), width*height);
            dataArr.push({data:bytes, genome_id: crossed_genome.id, node_genes:node_genes});
        }
        res.send(dataArr);
    });

    app.post('/enlarge', function(req, res){
        let params = req.body;
        if(params.color == 'true') {
            num_output = 3;
        }else{
            num_output = 1;
        }
        if(params.dense =='true') { 
            num_hidden_neurons = 64; 
        }else{
            num_hidden_neurons = 32; 
        }

        var enlarged_width = 980;
        var enlarged_height = 880;

        var selected_genome = params.selected_genome;
        selected_genome = Genome.reconstructGenome(selected_genome, num_output, num_hidden_neurons);
        res.on('finish', () => { console.log('Response sent.') });

        var inputs = createInputs(enlarged_width, enlarged_height);
        var outputs = selected_genome.evaluate([inputs[0], inputs[1], inputs[2]]);
        var data = outputs.output;
        console.log(data.shape)
            
        if(num_output==3) data = data.reshape([data.shape[0]*3])//RGB Channels
        var bytes = convertToByteArray(data.dataSync(), enlarged_height*enlarged_width, 'uint8');
        console.log(bytes.length)
        const canvas = createCanvas(enlarged_width, enlarged_height);
        var ctx = canvas.getContext('2d');
        var imageData = createImageData(bytes, enlarged_width, enlarged_height);
        console.log(enlarged_width, enlarged_height)
        ctx.putImageData(imageData, 0, 0);

        res.send(canvas.toDataURL())

    });

}

function convertToByteArray(data, size, type){
    var bytes = null;
    if(type && type=='uint8'){
        bytes = new Uint8ClampedArray(size * 4); //Convert to Uint8 array for enlarged images
    }else{
        bytes = new Array(size * 4); //Convert to Uint8 array on the UI
    }
    var r,g,b,a,j,i;
    for (i = 0; i < size; ++i) {
        r = void 0, g = void 0, b = void 0, a = void 0;
        if(num_output == 3){
            r = data[i * 3] * 255;
            g = data[i * 3 + 1] * 255;
            b = data[i * 3 + 2] * 255;
        }else{
            r = data[i] * 255;
            g = data[i] * 255;
            b = data[i] * 255;
        }
        a = 255;
        j = i * 4;
        bytes[j + 0] = Math.round(r);
        bytes[j + 1] = Math.round(g);
        bytes[j + 2] = Math.round(b);
        bytes[j + 3] = Math.round(a);
    }
    return bytes;
}
main();
