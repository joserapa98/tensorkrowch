Search.setIndex({docnames:["api","contents","index","network_components","node_operations","tn_models","usage"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,nbsphinx:4,sphinx:56},filenames:["api.rst","contents.rst","index.rst","network_components.rst","node_operations.rst","tn_models.rst","usage.rst"],objects:{"tensorkrowch.AbstractEdge":[[3,1,1,"","axes"],[3,1,1,"","axis1"],[3,1,1,"","axis2"],[3,2,1,"","contract_"],[3,2,1,"","is_attached_to"],[3,2,1,"","is_batch"],[3,2,1,"","is_dangling"],[3,1,1,"","name"],[3,1,1,"","node1"],[3,1,1,"","node2"],[3,1,1,"","nodes"],[3,2,1,"","size"],[3,2,1,"","svd_"]],"tensorkrowch.AbstractNode":[[3,1,1,"","axes"],[3,1,1,"","axes_names"],[3,2,1,"","contract_between"],[3,2,1,"","contract_between_"],[3,2,1,"","dim"],[3,2,1,"","disconnect"],[3,1,1,"","dtype"],[3,1,1,"","edges"],[3,2,1,"","get_axis"],[3,2,1,"","get_axis_num"],[3,2,1,"","get_edge"],[3,2,1,"","in_which_axis"],[3,2,1,"","is_data"],[3,2,1,"","is_leaf"],[3,2,1,"","is_node1"],[3,2,1,"","is_non_leaf"],[3,2,1,"","is_virtual"],[3,2,1,"","make_tensor"],[3,2,1,"","mean"],[3,2,1,"","move_to_network"],[3,1,1,"","name"],[3,2,1,"","neighbours"],[3,1,1,"","network"],[3,2,1,"","norm"],[3,2,1,"","param_edges"],[3,2,1,"","permute"],[3,1,1,"","rank"],[3,2,1,"","set_tensor"],[3,1,1,"","shape"],[3,2,1,"","size"],[3,2,1,"","split_"],[3,2,1,"","std"],[3,1,1,"","successors"],[3,2,1,"","sum"],[3,1,1,"","tensor"],[3,2,1,"","unset_tensor"]],"tensorkrowch.Axis":[[3,2,1,"","is_batch"],[3,2,1,"","is_node1"],[3,1,1,"","name"],[3,1,1,"","node"],[3,1,1,"","num"]],"tensorkrowch.Edge":[[3,2,1,"","change_size"],[3,2,1,"","connect"],[3,2,1,"","copy"],[3,2,1,"","dim"],[3,2,1,"","disconnect"],[3,2,1,"","parameterize"]],"tensorkrowch.ParamEdge":[[3,2,1,"","change_dim"],[3,2,1,"","change_size"],[3,2,1,"","compute_parameters"],[3,2,1,"","connect"],[3,2,1,"","copy"],[3,2,1,"","dim"],[3,2,1,"","disconnect"],[3,1,1,"","grad"],[3,2,1,"","make_matrix"],[3,1,1,"","matrix"],[3,1,1,"","module_name"],[3,2,1,"","parameterize"],[3,2,1,"","set_matrix"],[3,2,1,"","set_parameters"],[3,1,1,"","shift"],[3,2,1,"","sigmoid"],[3,1,1,"","slope"]],"tensorkrowch.ParamNode":[[3,2,1,"","copy"],[3,1,1,"","grad"],[3,2,1,"","parameterize"]],"tensorkrowch.ParamStackEdge":[[3,2,1,"","connect"],[3,1,1,"","edges"],[3,1,1,"","matrix"],[3,1,1,"","node1_lists"]],"tensorkrowch.ParamStackNode":[[3,1,1,"","edges_dict"],[3,1,1,"","node1_lists_dict"]],"tensorkrowch.StackEdge":[[3,2,1,"","connect"],[3,1,1,"","edges"],[3,1,1,"","node1_lists"]],"tensorkrowch.StackNode":[[3,1,1,"","edges_dict"],[3,1,1,"","node1_lists_dict"]],"tensorkrowch.TensorNetwork":[[3,2,1,"","clear"],[3,2,1,"","contract"],[3,1,1,"","data_nodes"],[3,2,1,"","delete_node"],[3,1,1,"","edges"],[3,2,1,"","forward"],[3,2,1,"","initialize"],[3,1,1,"","leaf_nodes"],[3,1,1,"","nodes"],[3,1,1,"","non_leaf_nodes"],[3,2,1,"","parameterize"],[3,2,1,"","set_data_nodes"],[3,2,1,"","trace"],[3,1,1,"","virtual_nodes"]],"tensorkrowch.node_operations":[[6,3,1,"","_contract_edges_first"]],tensorkrowch:[[3,0,1,"","AbstractEdge"],[3,0,1,"","AbstractNode"],[3,0,1,"","Axis"],[5,0,1,"","ConvMPS"],[5,0,1,"","ConvMPSLayer"],[5,0,1,"","ConvPEPS"],[5,0,1,"","ConvTree"],[5,0,1,"","ConvUMPS"],[5,0,1,"","ConvUMPSLayer"],[5,0,1,"","ConvUPEPS"],[5,0,1,"","ConvUTree"],[3,0,1,"","Edge"],[5,0,1,"","MPS"],[5,0,1,"","MPSLayer"],[6,0,1,"","Node"],[4,0,1,"","Operation"],[5,0,1,"","PEPS"],[3,0,1,"","ParamEdge"],[3,0,1,"","ParamNode"],[3,0,1,"","ParamStackEdge"],[3,0,1,"","ParamStackNode"],[3,0,1,"","StackEdge"],[3,0,1,"","StackNode"],[3,0,1,"","Successor"],[3,0,1,"","TensorNetwork"],[5,0,1,"","Tree"],[5,0,1,"","UMPS"],[5,0,1,"","UMPSLayer"],[5,0,1,"","UPEPS"],[5,0,1,"","UTree"],[4,3,1,"","add"],[4,3,1,"","connect"],[4,3,1,"","connect_stack"],[4,3,1,"","contract_between"],[4,3,1,"","contract_edges"],[4,3,1,"","disconnect"],[4,3,1,"","einsum"],[4,3,1,"","mul"],[4,3,1,"","permute"],[4,3,1,"","split"],[4,3,1,"","stack"],[4,3,1,"","stacked_einsum"],[4,3,1,"","sub"],[4,3,1,"","tprod"],[4,3,1,"","unbind"]]},objnames:{"0":["py","class","Python class"],"1":["py","property","Python property"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:class","1":"py:property","2":"py:method","3":"py:function"},terms:{"0":[3,5],"03366":3,"1":[3,5],"10":3,"2":[3,5,6],"2203":3,"3":[3,6],"4":3,"5":3,"abstract":3,"boolean":[3,5,6],"case":[3,6],"class":[0,3,5,6],"default":3,"do":3,"final":3,"float":[3,6],"function":3,"import":[3,6],"int":[3,6],"new":[3,6],"return":[3,4,6],"static":3,"true":[3,6],A:3,At:3,By:3,For:[3,6],If:[3,5,6],In:[3,5],It:[3,6],That:3,The:[2,3,4],Then:3,These:3,To:[3,6],_:3,_contract_edges_first:6,_description_:3,_list:3,_tensor_info:3,_type_:3,ab:3,abl:3,about:3,abov:3,abstractedg:[3,6],abstractnod:[3,6],access:[3,6],accord:3,activ:2,ad:[3,4],adapt:4,add:4,addit:3,advanc:3,after:3,algorithm:3,all:[3,4,5,6],allow:3,almost:3,alpha_1:3,alpha_n:3,alreadi:[3,4],also:[3,4,6],although:3,alwai:[3,5],an:[2,3,4,5,6],ani:[2,3,6],anoth:[3,6],api:1,appli:4,applic:3,ar:[3,4,5,6],arg:[3,4],argument:[3,6],arxiv:3,attach:[3,6],attribut:[3,6],automat:[3,6],automemori:3,auxiliari:3,avoid:3,ax:[3,4,6],axes_nam:[3,6],axi:[0,4,6],axis1:3,axis2:3,b:3,b_:3,base:[3,6],basic:3,batch:[3,4,5,6],been:3,befor:[3,4],being:[2,3],belong:[3,6],besid:3,beta_1:3,beta_m:3,beta_n:3,between:[3,4,6],blank:[3,6],bond:[3,5],bool:[3,6],both:[3,4],boundari:5,build:[2,3],built:[2,4],cada:3,calcul:3,call:3,can:[3,6],cannot:[3,6],carri:3,cast:3,caus:3,cdot:3,certain:[3,6],chang:3,change_dim:3,change_s:3,charact:[3,4],check_first:4,child:3,chosen:3,clear:3,close:3,code:6,coincid:3,collect:3,column:5,come:[3,4],complet:[3,6],compon:[0,1],comput:[3,6],compute_paramet:3,condit:5,conform:3,connect:[0,3,6],connect_stack:4,construct:3,contain:[3,6],contract:[3,4,5,6],contract_:3,contract_between:[3,4],contract_between_:3,contract_edg:4,contrat:5,convers:3,convmp:5,convmpslay:5,convpep:5,convtre:5,convump:5,convumpslay:5,convupep:5,convutre:5,copi:[3,6],copy_:3,correct:3,correspond:[3,4],could:3,coupl:4,cpu:3,creat:[3,5,6],crop:3,cum_percentag:3,current:3,custom:3,cutoff:3,d:3,d_:3,d_bond:5,d_phy:5,dangl:[3,4],data:[3,5,6],data_nod:3,de:[3,6],deep:2,defin:3,delete_nod:3,describ:3,desir:3,determin:3,develop:2,devic:[3,6],df:3,diagon:3,dictionari:3,differ:[3,6],dilat:5,dim:3,dimens:[3,5],directli:3,disconnect:[0,3],distinct:[3,6],doe:3,done:3,drive:3,dtype:3,dure:3,e:[3,6],each:[3,4,5,6],easi:2,easili:3,edg:[0,5,6],edge1:4,edge2:4,edge_nodea_right_nodeb_left:3,edgea:3,edgeb:3,edges_dict:3,effect:3,effici:3,einsum:4,element:[3,4,5],elimina:3,empti:3,enabl:[2,3,4],enough:3,ensur:3,entir:3,equal:3,equat:3,error:3,essenti:3,etc:3,even:3,everi:3,exact:3,exampl:3,execut:3,expand:3,expect:3,experi:3,experiment:3,explain:3,explicitli:3,extend:3,fals:[3,5,6],faster:3,featur:3,feed:3,fill:3,first:[3,4,6],fix:[3,6],follow:3,form:[3,4,5],forward:3,framework:2,from:[3,4,6],front:3,fulfil:3,func1:4,func2:4,func:3,further:3,furthermor:3,g:[3,6],gamma:3,gener:[2,3],get_axi:3,get_axis_num:3,get_edg:3,given:[3,5],go:3,grad:3,gradient:3,graph:3,greater:3,grid:5,ha:[3,6],had:3,have:[3,5],help:3,henc:[3,6],here:[3,6],hidden:3,high:3,higli:2,hint:3,horizont:5,how:3,howev:3,http:3,i:[3,5],ident:3,implement:3,in_channel:5,in_which_axi:3,includ:[2,3,5],inde:[3,6],independ:3,index:3,indic:[3,5,6],infer:3,info:3,inform:3,inherit:[3,6],init_method:[3,6],initi:[3,6],inlin:5,inline_input:5,inline_mat:5,inplac:3,input:[3,4,5],input_edg:3,instal:1,instanc:3,instanti:3,instead:[3,6],intend:[3,6],intermedi:[3,6],investig:3,involv:[3,4],is_attached_to:3,is_batch:3,is_dangl:3,is_data:3,is_leaf:3,is_node1:3,is_non_leaf:3,is_virtu:3,iter:3,its:[3,6],itself:3,j:3,just:3,keep:3,kei:3,kept:3,kernel_s:5,keyword:[3,6],kind:3,know:3,kwarg:[3,4,6],l_posit:5,label:5,larger:3,largest:3,last:5,layer:5,ldot:3,leaf:[3,6],leaf_nod:3,learn:[2,3],learnabl:3,left:3,leg:3,length:3,level:3,li:3,librari:2,like:[0,3,6],link:3,list:[3,4,5,6],lo:3,longer:3,loss:3,low:3,machin:2,made:[3,6],main:3,mainli:3,make:[2,3],make_matrix:3,make_tensor:[3,6],mandatori:3,mani:3,match:3,matric:5,matrix:3,mean:3,memori:3,memoria:3,method:[3,6],might:3,min:3,minimum:3,mode:3,model:[0,1,2,3],modif:3,modifi:3,modul:[3,5],module_nam:3,more:3,most:3,move:3,move_nam:3,move_to_network:3,mp:[0,3],mpslayer:0,mul:4,must:[3,6],my_stack:3,mybatch:3,n:3,n_col:5,n_featur:3,n_label:5,n_row:5,n_site:5,name:[3,4,6],names_batch_edg:3,necessari:3,need:3,neighbour:3,neither:[3,6],net:3,network:[0,1,2,6],never:3,new_edg:3,new_edgea:3,new_edgeb:3,new_nod:6,new_paramedg:3,new_siz:3,next:3,nn:3,node1:[3,4,6],node1_ax:3,node1_list:[3,6],node1_lists_dict:3,node2:[3,4,6],node2_ax:3,node:[0,1,5,6],node_1:3,node_2:3,node_oper:6,node_ref:3,nodea:3,nodeb:3,nodes_list:4,nodo:3,non:[3,4,6],non_leaf:3,non_leaf_nod:3,none:[3,4,5,6],nor:[3,6],norm:3,normal:3,noth:3,num:3,num_batch:5,num_batch_edg:3,number:[3,5],obc:5,object:3,occur:3,one:[3,5,6],ones:[3,6],onli:[3,4,5],open:5,oper:[0,1,3,6],optim:[2,3],option:[3,6],order:3,org:3,origin:[3,6],other:[3,4,6],otherwis:[3,6],our:3,out:3,out_channel:5,output:[4,5],output_nod:5,over:3,overload:3,overrid:[3,6],override_edg:[3,6],override_nod:[3,6],overriden:[3,6],own:3,p:3,pad:5,pair:3,pairwis:5,paper:3,parallel:3,param:3,param_bond:5,param_edg:[3,6],param_nod:3,paramedg:[3,6],paramet:[3,4,5,6],parameter:[3,6],parametr:[3,5],paramnod:[3,6],paramstackedg:3,paramstacknod:3,part:3,particular:3,pep:0,perform:[3,4],period:5,permut:[3,4],physic:5,pip:6,pipelin:2,place:3,plai:3,point:3,poner:3,posit:5,possibl:3,prefix:3,prev_siz:3,print:3,process:3,project:2,properti:[3,6],propia:3,provid:[3,6],purpos:2,put:3,python:2,pytorch:[2,3],rais:3,rand:[3,6],randn:[3,6],rang:3,rank:3,rather:3,re:3,realli:3,reattach:[3,6],reduc:3,redund:3,refer:[1,3],referenc:3,regard:3,regim:3,regular:3,relev:3,remov:3,replac:[3,6],requir:[3,6],respect:3,rest:3,result:[3,4,6],retriev:3,right:[3,5],role:3,row:5,s:[3,6],sai:3,same:[3,4,5,6],save:3,second:[3,6],see:[3,6],sepcifi:3,sequenc:[3,5,6],set:[3,4],set_data_nod:3,set_matrix:3,set_param:3,set_paramet:3,set_tensor:3,sever:3,shape:[3,5,6],share:[3,4,6],shift:3,should:[3,5,6],shrink:3,side:3,sigma:3,sigmoid:3,similar:3,sinc:3,singl:5,site:[3,5],sites_per_lay:5,size:[3,6],skip:3,slide:3,slope:3,smaller:3,snippet:6,so:3,some:3,sort:3,sourc:[3,4,5,6],space:[3,6],special:[3,4],specif:3,specifi:3,split:[3,4],split_:3,squar:3,stack:[3,4,5],stack_data:3,stack_nod:3,stacked_einsum:4,stackedg:3,stacknod:[3,6],stai:3,std:3,step:3,stick:3,store:[3,6],str:[3,6],stride:5,string:[4,5],structur:3,su:3,sub:4,subclass:[3,6],submodul:3,successor:0,suit:3,sum:3,sum_:3,svd:3,svd_:3,symbol:3,t:3,t_:3,taken:3,tensor:[0,1,2,6],tensorkrowch:[3,4,5,6],tensornetwork:[3,6],text:3,th:5,than:3,thei:[3,6],them:[2,3],therefor:3,thi:[2,3,6],though:3,through:3,thu:3,time:[3,6],tk:[3,6],tn:3,todo:3,top:[2,3],topolog:3,torch:[3,6],toward:3,tprod:4,trace:3,track:3,train:[2,3,6],trainabl:[3,6],tree:0,tupl:[3,6],turn:3,two:[3,4,6],type:[3,4,6],ump:5,umpslay:5,unbind:[3,4],unbind_0:3,under:2,uniform:3,uniqu:3,unset_tensor:3,until:3,updat:3,upep:5,us:[2,3,5,6],usag:[1,3],user:3,usual:3,utre:5,valu:3,valueerror:3,variat:3,variou:5,verifi:3,version:3,vertic:5,via:3,virtual:[3,6],virtual_nod:3,visit:3,vuelv:3,wa:3,wai:3,we:[3,4,5],were:[3,4],what:3,when:[3,6],where:[3,6],whether:[3,5,6],which:[3,4],whole:3,whose:3,wise:3,without:3,word:3,work:3,worri:3,would:[3,6],wrap:3,x:3,y:3,you:6,zero:[3,6]},titles:["API Reference","&lt;no title&gt;","TensorKrowch documentation","Tensor Network Components","Node Operations","Models","Usage"],titleterms:{"class":4,api:0,axi:3,cite:2,compon:3,connect:4,content:1,disconnect:4,document:2,edg:[3,4],exampl:2,guid:2,instal:6,like:4,model:5,mp:5,mpslayer:5,network:3,node:[3,4],oper:4,pep:5,refer:0,successor:3,tensor:[3,4],tensorkrowch:2,tree:5,usag:6,user:2}})