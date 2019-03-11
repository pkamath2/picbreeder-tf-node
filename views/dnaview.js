function findColor(f){
    if(f.indexOf('output')>-1) return 'black';
    if(f.indexOf('sin')>-1) return 'blue';
    if(f.indexOf('tanh')>-1) return 'yellow';
    if(f.indexOf('sigmoid')>-1) return 'red';
    if(f.indexOf('softplus')>-1) return 'pink';
    if(f.indexOf('gaussian')>-1) return 'green';
    if(f.indexOf('psychedelic')>-1) return 'orange';
    return 'white';
}

/*
 *  Developed with the help of below three examples on bl.ocks - 
 *  D3.js v4 Force Directed Graph with Labels: https://bl.ocks.org/heybignick/3faf257bbbbc7743bb72310d03b86ee8
 *  Force-Directed Graph: https://bl.ocks.org/mbostock/4062045
 *  Bounding Box: https://bl.ocks.org/puzzler10/2531c035e8d514f125c4d15433f79d74
 */
function drawDNA(selected_thumbnails){
    selected_thumbnails.forEach((t,ind) => {
        let svg = null;
        if(ind==0) svg = d3.select('#dna_1_svg');
        if(ind==1) svg = d3.select('#dna_2_svg');
        svg.selectAll("*").remove();

        let nodes = [];
        let links = [];
        let node_genes = thumbnail_nodegene_map.get(t);
        console.log(node_genes)
        node_genes.forEach(node_gene => {
            nodes.push({'id':node_gene.innovation_number, 'group':node_gene.activation, 'name':node_gene.name+"_"+node_gene.activation});
            node_gene.from_conn_arr.forEach(c => {
                links.push({'source':c, 'target':node_gene.innovation_number, 'value':1, 'activation': node_gene.activation});
            });
        });
        links = links.map(d => Object.create(d));
        nodes = nodes.map(d => Object.create(d));
        const simulation = d3.forceSimulation(nodes)
                         .force("collide", d3.forceCollide(50))
                        .force("link", d3.forceLink(links).id(d => d.id));
        nodes.forEach(n => {
                if(n.name.indexOf('input_0')>-1){n.fx=100; n.fy=50;} 
                if(n.name.indexOf('input_1')>-1){n.fx=150; n.fy=50;} 
                if(n.name.indexOf('input_2')>-1){n.fx=200; n.fy=50;} 
                if(n.name.indexOf('output')>-1){n.fx=150; n.fy=270;} 
        });
        const link = svg.append("g").selectAll("line")
                        .data(links).enter()
                        .append("line")
                        .attr("stroke", "#999")
                        .attr("stroke-opacity", 0.6)
                        .attr("stroke-width", d => 1);

        console.log(nodes)
        const g_node = svg.append("g").selectAll("g").data(nodes).enter().append("g");

        const node = g_node
                        .append("circle")
                        .attr("stroke", "#000")
                        .attr("stroke-width", 1.5)
                        .attr("r", 5)
                        .attr("fill", d=>findColor(d.name));
                                        
                
        const text_node = g_node
                            .append('text')
                            .text(d=>{if(d.group.indexOf(".")>-1) return d.group.split(".")[1]; else return d.group;})
                            .attr('x', d => 6)
                            .attr('y', d => 6)
                            .attr('class','small');

        simulation.on("tick", () => {
                            link.attr("x1", d => Math.max(5, Math.min(250, d.source.x)))
                                .attr("y1", d => Math.max(5, Math.min(250, d.source.y)))
                                .attr("x2", d => Math.max(5, Math.min(250, d.target.x)))
                                .attr("y2", d => Math.max(5, Math.min(250, d.target.y)));
                            
                            g_node.attr('transform', function(d) {
                                return "translate(" + Math.max(5, Math.min(250, d.x)) + "," + Math.max(5, Math.min(250, d.y)) + ")";//Bounding Box
                            });
        });
    });
}