n=10;// Nombre de pages
  
function show_adj(Adj,diameters)
  [lhs,rhs]=argn(0); 
  if rhs < 2 then diameters=30*ones(1,n);end
  graph=mat_2_graph(sparse(Adj),1,'node-node'); 
  graph('node_x')=300*cos(2*%pi*(1:n)/(n+1));
  graph('node_y')=300*sin(2*%pi*(1:n)/(n+1)); 
  graph('node_name')=string([1:n]);
  graph('node_diam')=diameters; 
  //graph('node_color')= 1:n;
  //show_graph(graph);
  rep=[1,1,1,1,2,2,2,2,2,2,2,2,2]; 
  plot_graph(graph,rep);
endfunction

Adj=grand(n,n,'bin',1,0.2);show_adj(Adj);