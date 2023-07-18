Search.setIndex({docnames:["api","components","contents","embeddings","index","initializers","installation","models","operations","tutorials","tutorials/0_first_steps","tutorials/1_creating_tensor_network","tutorials/2_contracting_tensor_network","tutorials/3_memory_management","tutorials/4_types_of_nodes","tutorials/5_subclass_tensor_network","tutorials/6_mix_with_pytorch"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,nbsphinx:4,sphinx:56},filenames:["api.rst","components.rst","contents.rst","embeddings.rst","index.rst","initializers.rst","installation.rst","models.rst","operations.rst","tutorials.rst","tutorials/0_first_steps.rst","tutorials/1_creating_tensor_network.rst","tutorials/2_contracting_tensor_network.rst","tutorials/3_memory_management.rst","tutorials/4_types_of_nodes.rst","tutorials/5_subclass_tensor_network.rst","tutorials/6_mix_with_pytorch.rst"],objects:{"tensorkrowch.AbstractNode":[[1,1,1,"","axes"],[1,1,1,"","axes_names"],[1,2,1,"","contract_between"],[1,2,1,"","contract_between_"],[1,1,1,"","device"],[1,2,1,"","disconnect"],[1,1,1,"","dtype"],[1,1,1,"","edges"],[1,2,1,"","get_axis"],[1,2,1,"","get_axis_num"],[1,2,1,"","get_edge"],[1,2,1,"","in_which_axis"],[1,2,1,"","is_data"],[1,2,1,"","is_leaf"],[1,2,1,"","is_node1"],[1,2,1,"","is_resultant"],[1,2,1,"","is_virtual"],[1,2,1,"","make_tensor"],[1,2,1,"","mean"],[1,2,1,"","move_to_network"],[1,1,1,"","name"],[1,2,1,"","neighbours"],[1,1,1,"","network"],[1,2,1,"","norm"],[1,2,1,"","permute"],[1,2,1,"","permute_"],[1,1,1,"","rank"],[1,2,1,"","reattach_edges"],[1,2,1,"","reset_tensor_address"],[1,2,1,"","set_tensor"],[1,2,1,"","set_tensor_from"],[1,1,1,"","shape"],[1,2,1,"","size"],[1,2,1,"","split"],[1,2,1,"","split_"],[1,2,1,"","std"],[1,1,1,"","successors"],[1,2,1,"","sum"],[1,1,1,"","tensor"],[1,2,1,"","tensor_address"],[1,2,1,"","unset_tensor"]],"tensorkrowch.Axis":[[1,2,1,"","is_batch"],[1,2,1,"","is_node1"],[1,1,1,"","name"],[1,1,1,"","node"],[1,1,1,"","num"]],"tensorkrowch.Edge":[[1,1,1,"","axes"],[1,1,1,"","axis1"],[1,1,1,"","axis2"],[1,2,1,"","change_size"],[1,2,1,"","connect"],[1,2,1,"","contract_"],[1,2,1,"","copy"],[1,2,1,"","disconnect"],[1,2,1,"","is_attached_to"],[1,2,1,"","is_batch"],[1,2,1,"","is_dangling"],[1,1,1,"","name"],[1,1,1,"","node1"],[1,1,1,"","node2"],[1,1,1,"","nodes"],[1,2,1,"","qr_"],[1,2,1,"","rq_"],[1,2,1,"","size"],[1,2,1,"","svd_"],[1,2,1,"","svdr_"]],"tensorkrowch.Node":[[1,2,1,"","copy"],[1,2,1,"","parameterize"]],"tensorkrowch.ParamNode":[[1,2,1,"","copy"],[1,1,1,"","grad"],[1,2,1,"","parameterize"]],"tensorkrowch.ParamStackNode":[[1,1,1,"","edges_dict"],[1,1,1,"","node1_lists_dict"]],"tensorkrowch.StackEdge":[[1,2,1,"","connect"],[1,1,1,"","edges"],[1,1,1,"","node1_list"]],"tensorkrowch.StackNode":[[1,1,1,"","edges_dict"],[1,1,1,"","node1_lists_dict"]],"tensorkrowch.TensorNetwork":[[1,2,1,"","add_data"],[1,1,1,"","auto_stack"],[1,1,1,"","auto_unbind"],[1,2,1,"","contract"],[1,2,1,"","copy"],[1,1,1,"","data_nodes"],[1,2,1,"","delete_node"],[1,1,1,"","edges"],[1,2,1,"","forward"],[1,1,1,"","leaf_nodes"],[1,1,1,"","nodes"],[1,1,1,"","nodes_names"],[1,2,1,"","parameterize"],[1,2,1,"","reset"],[1,1,1,"","resultant_nodes"],[1,2,1,"","set_data_nodes"],[1,2,1,"","trace"],[1,2,1,"","unset_data_nodes"],[1,1,1,"","virtual_nodes"]],"tensorkrowch.embeddings":[[3,3,1,"","add_ones"],[3,3,1,"","poly"],[3,3,1,"","unit"]],"tensorkrowch.models":[[7,0,1,"","ConvMPS"],[7,0,1,"","ConvMPSLayer"],[7,0,1,"","ConvPEPS"],[7,0,1,"","ConvTree"],[7,0,1,"","ConvUMPS"],[7,0,1,"","ConvUMPSLayer"],[7,0,1,"","ConvUPEPS"],[7,0,1,"","ConvUTree"],[7,0,1,"","MPS"],[7,0,1,"","MPSLayer"],[7,0,1,"","PEPS"],[7,0,1,"","Tree"],[7,0,1,"","UMPS"],[7,0,1,"","UMPSLayer"],[7,0,1,"","UPEPS"],[7,0,1,"","UTree"]],"tensorkrowch.models.ConvMPS":[[7,1,1,"","dilation"],[7,2,1,"","forward"],[7,1,1,"","in_channels"],[7,1,1,"","kernel_size"],[7,1,1,"","padding"],[7,1,1,"","stride"]],"tensorkrowch.models.ConvMPSLayer":[[7,1,1,"","dilation"],[7,2,1,"","forward"],[7,1,1,"","in_channels"],[7,1,1,"","kernel_size"],[7,1,1,"","padding"],[7,1,1,"","stride"]],"tensorkrowch.models.ConvPEPS":[[7,1,1,"","dilation"],[7,2,1,"","forward"],[7,1,1,"","in_channels"],[7,1,1,"","kernel_size"],[7,1,1,"","padding"],[7,1,1,"","stride"]],"tensorkrowch.models.ConvTree":[[7,1,1,"","dilation"],[7,2,1,"","forward"],[7,1,1,"","in_channels"],[7,1,1,"","kernel_size"],[7,1,1,"","padding"],[7,1,1,"","stride"]],"tensorkrowch.models.ConvUMPS":[[7,1,1,"","dilation"],[7,2,1,"","forward"],[7,1,1,"","in_channels"],[7,1,1,"","kernel_size"],[7,1,1,"","padding"],[7,1,1,"","stride"]],"tensorkrowch.models.ConvUMPSLayer":[[7,1,1,"","dilation"],[7,2,1,"","forward"],[7,1,1,"","in_channels"],[7,1,1,"","kernel_size"],[7,1,1,"","padding"],[7,1,1,"","stride"]],"tensorkrowch.models.ConvUPEPS":[[7,1,1,"","dilation"],[7,2,1,"","forward"],[7,1,1,"","in_channels"],[7,1,1,"","kernel_size"],[7,1,1,"","padding"],[7,1,1,"","stride"]],"tensorkrowch.models.ConvUTree":[[7,1,1,"","dilation"],[7,2,1,"","forward"],[7,1,1,"","in_channels"],[7,1,1,"","kernel_size"],[7,1,1,"","padding"],[7,1,1,"","stride"]],"tensorkrowch.models.MPS":[[7,1,1,"","bond_dim"],[7,1,1,"","boundary"],[7,2,1,"","canonicalize"],[7,2,1,"","canonicalize_univocal"],[7,2,1,"","contract"],[7,1,1,"","in_dim"],[7,2,1,"","initialize"],[7,1,1,"","n_batches"],[7,1,1,"","n_features"],[7,2,1,"","set_data_nodes"]],"tensorkrowch.models.MPSLayer":[[7,1,1,"","bond_dim"],[7,1,1,"","boundary"],[7,2,1,"","canonicalize"],[7,2,1,"","canonicalize_univocal"],[7,2,1,"","contract"],[7,1,1,"","in_dim"],[7,2,1,"","initialize"],[7,1,1,"","n_batches"],[7,1,1,"","n_features"],[7,1,1,"","out_dim"],[7,1,1,"","out_position"],[7,1,1,"","phys_dim"],[7,2,1,"","set_data_nodes"]],"tensorkrowch.models.PEPS":[[7,1,1,"","bond_dim"],[7,1,1,"","boundary"],[7,2,1,"","contract"],[7,1,1,"","in_dim"],[7,2,1,"","initialize"],[7,1,1,"","n_batches"],[7,1,1,"","n_cols"],[7,1,1,"","n_rows"],[7,2,1,"","set_data_nodes"]],"tensorkrowch.models.Tree":[[7,1,1,"","bond_dim"],[7,2,1,"","canonicalize"],[7,2,1,"","contract"],[7,2,1,"","initialize"],[7,1,1,"","n_batches"],[7,2,1,"","set_data_nodes"],[7,1,1,"","sites_per_layer"]],"tensorkrowch.models.UMPS":[[7,1,1,"","bond_dim"],[7,2,1,"","contract"],[7,1,1,"","in_dim"],[7,2,1,"","initialize"],[7,1,1,"","n_batches"],[7,1,1,"","n_features"],[7,2,1,"","set_data_nodes"]],"tensorkrowch.models.UMPSLayer":[[7,1,1,"","bond_dim"],[7,1,1,"","boundary"],[7,2,1,"","contract"],[7,1,1,"","in_dim"],[7,2,1,"","initialize"],[7,1,1,"","n_batches"],[7,1,1,"","n_features"],[7,1,1,"","out_dim"],[7,1,1,"","out_position"],[7,2,1,"","set_data_nodes"]],"tensorkrowch.models.UPEPS":[[7,1,1,"","bond_dim"],[7,2,1,"","contract"],[7,1,1,"","in_dim"],[7,2,1,"","initialize"],[7,1,1,"","n_batches"],[7,1,1,"","n_cols"],[7,1,1,"","n_rows"],[7,2,1,"","set_data_nodes"]],"tensorkrowch.models.UTree":[[7,1,1,"","bond_dim"],[7,2,1,"","contract"],[7,2,1,"","initialize"],[7,1,1,"","n_batches"],[7,2,1,"","set_data_nodes"],[7,1,1,"","sites_per_layer"]],tensorkrowch:[[1,0,1,"","AbstractNode"],[1,0,1,"","Axis"],[1,0,1,"","Edge"],[1,0,1,"","Node"],[8,0,1,"","Operation"],[1,0,1,"","ParamNode"],[1,0,1,"","ParamStackNode"],[1,0,1,"","StackEdge"],[1,0,1,"","StackNode"],[1,0,1,"","Successor"],[1,0,1,"","TensorNetwork"],[8,3,1,"","add"],[8,3,1,"","connect"],[8,3,1,"","connect_stack"],[8,3,1,"","contract_"],[8,3,1,"","contract_between"],[8,3,1,"","contract_between_"],[8,3,1,"","contract_edges"],[5,3,1,"","copy"],[8,3,1,"","disconnect"],[8,3,1,"","einsum"],[5,3,1,"","empty"],[8,3,1,"","mul"],[5,3,1,"","ones"],[8,3,1,"","permute"],[8,3,1,"","permute_"],[8,3,1,"","qr_"],[5,3,1,"","rand"],[5,3,1,"","randn"],[8,3,1,"","rq_"],[8,3,1,"","split"],[8,3,1,"","split_"],[8,3,1,"","stack"],[8,3,1,"","stacked_einsum"],[8,3,1,"","sub"],[8,3,1,"","svd_"],[8,3,1,"","svdr_"],[8,3,1,"","tprod"],[8,3,1,"","unbind"],[5,3,1,"","zeros"]]},objnames:{"0":["py","class","Python class"],"1":["py","property","Python property"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:class","1":"py:property","2":"py:method","3":"py:function"},terms:{"0":[1,3,4,5,7,8,10,12,14,15,16],"00":3,"0000e":3,"0086":1,"0101":1,"0188":1,"0234":1,"0371":1,"0412":1,"0440":1,"0445":1,"0465":1,"0501":1,"0521":1,"0633":1,"0704":1,"0716":10,"0743":1,"0751":10,"08":3,"0806":1,"0815":10,"0836":10,"08595":4,"09":7,"0904":10,"1":[1,3,4,5,7,8,16],"10":[1,4,7,8,10,12,15,16],"100":[1,3,4,8,11,12,13,14],"10000":[10,16],"1018":10,"1024":[10,16],"1046":10,"1091":1,"110753":16,"1135":10,"1211":1,"1317":10,"136940":16,"1371":1,"14":[4,6,10,11,16],"1496":1,"15":[1,8,12],"150710":10,"1642":10,"1654":1,"1998":1,"1e":[7,10,15,16],"2":[1,3,7,8,16],"20":[1,7,8,16],"2013":16,"2066":10,"2083":16,"2091":1,"2095":1,"2111":1,"2254":1,"2257":16,"2306":4,"2316":1,"235660":10,"2380":1,"2390":1,"2449":1,"2461":1,"2463":16,"2477":1,"2503":1,"2517":1,"2694":16,"2730":1,"2775":1,"2799":1,"28":[10,16],"2808":1,"2856":1,"2866":1,"2898":1,"2921":10,"2924":16,"2d":7,"3":[1,3,4,7,8,14,16],"30":16,"3088":1,"3139":1,"32":7,"3222":1,"3245":16,"3340":1,"3370":1,"3371":1,"3381":1,"3393":1,"3427":1,"3489":1,"3508":1,"3513":1,"3585":1,"36":10,"3711e":3,"3760":1,"3784":1,"3821":1,"3824":16,"4":[1,3,7,8,12,14,16],"40":16,"4005":1,"4181":1,"4184":1,"4216":1,"4225":16,"4383":1,"4385":1,"4402":1,"4431":1,"4461":1,"4572":1,"4588":1,"4731":1,"4761":1,"4974":1,"4f":[10,16],"5":[1,3,4,7,8,11,12,13,14,16],"50":16,"500":[4,10,15,16],"5021":1,"5029":1,"5069":1,"5161":1,"5224":16,"5401":1,"5406":1,"5567":1,"5570":1,"5731":1,"5749":10,"5760":1,"5797":1,"5920":1,"6":[7,16],"60":16,"60000":[10,16],"6023":1,"6225":1,"6227":1,"6295":1,"6356":1,"6399":1,"6492":1,"6495":1,"6524":1,"6545":1,"6551":1,"6752":10,"6811":1,"6925":1,"6982":1,"7":[1,4,7,8,12,13,16],"70":16,"7592":1,"7688":1,"7752":1,"7812":1,"7997":1,"8":[4,7,10],"80":16,"8090":1,"8147":1,"8227":1,"8361":1,"8387":1,"8441":1,"8442":1,"8570":16,"8599":16,"8633":1,"8649":1,"8758":16,"8814":16,"8815":1,"8829":16,"8859":1,"8860":16,"8884":16,"8908":16,"8911":1,"8915":16,"8924":10,"8958":16,"8969":16,"8979":16,"8990":10,"8993":16,"9":[4,10,15],"90":16,"9006":1,"9078":16,"9122":10,"9145":1,"9163":16,"9194":10,"9219":16,"9226":16,"9265":1,"9320":1,"9360":10,"9378":10,"9432":1,"9443":10,"9456":10,"9502":10,"9551":1,"9561":1,"9595":10,"9601":10,"9618":1,"9632":10,"9654":10,"9655":10,"9669":10,"9684":10,"9687":10,"9693":10,"9699":10,"9720":10,"9721":10,"9723":10,"9725":10,"9740":10,"9745":10,"9748":10,"9764":10,"9777":10,"98":[10,16],"9844":1,"9925":1,"9942":1,"9957":1,"abstract":1,"boolean":[1,5,7],"byte":[10,16],"case":[1,13,14,15],"class":[0,1,7,10,12,15,16],"default":[1,3,13,14],"do":[1,10,11,13,14,15],"final":[1,12,15],"float":[1,5,7,8],"function":[1,8,12,16],"garc\u00eda":4,"import":[1,4,11,12,13,14,15,16],"int":[1,3,5,7,8],"n\u00ba":[10,16],"new":[1,3,5,7,8,10,11,12,13,15,16],"p\u00e9rez":4,"return":[1,3,5,7,8,10,11,12,13,15,16],"super":[1,15,16],"true":[1,4,5,8,10,12,13,14,15,16],"try":[12,16],"while":[1,4,12,13,14],A:[1,4,7,8,11],And:[13,16],As:[4,10,11,12,13],At:[1,11],Be:[1,12],But:[12,13,15,16],By:[1,3,4,10,11],For:[1,8,11,12,13,15],If:[1,3,4,5,7,8,11,13,15],In:[1,4,8,10,11,12,13,14,15,16],Is:7,It:[1,4,5,7,8,10,11,12,16],Its:[1,8],Not:15,Of:[13,15],On:1,One:[1,11],That:[1,3,7,8,12,13,15,16],The:[1,3,4,7,8,9,11,12,13,16],Then:[1,13],There:[1,12,14,15],These:[1,4,7,11,12,14],To:[1,4,6,8,10,11,12,13,14,15],With:[1,4,8,11,12,15],_:[1,8,10,11,15,16],__call__:1,__init__:[1,15,16],_channel:7,_copi:1,_dim:1,_percentag:[1,7,8],_size:[3,7],_size_0:7,_size_1:7,_size_:1,a_:8,abil:4,abl:[1,11,12,15],about:[1,8,11,12,13,14,15],abov:[1,11,15],abstractnod:[0,8],acc:[10,16],acceler:1,accept:12,access:[1,5,11,12,13],accomplish:[1,14],accord:[1,8],account:[1,14],accuraci:[10,16],achiev:1,act:[1,11,13],action:1,activ:1,actual:[1,7,11,12,13,16],ad:[1,3,8,12,15],adam:[10,16],adapt:8,add:[0,1,10,12,15,16],add_data:[1,14,15],add_on:[0,16],addit:[1,8],addition:1,address:[1,11,13],advanc:[1,4,9],advantag:[13,15],affect:8,after:[1,7,10,16],afterward:[1,8],again:[10,13,15],aim:10,alejandro:4,algorithm:[1,7,13,15,16],all:[1,5,7,8,10,11,12,13,14,15,16],allow:[1,7],almost:[1,12,13,14,16],along:[1,8],alreadi:[1,8,10,11,12,15],also:[1,4,6,7,8,10,11,12,13,14,15],although:[1,11,12,13,14,15],alwai:[1,4,7,8,10,11,12,13,14,15],amount:[8,10,16],an:[1,4,7,8,10,11,12,13,14,15],ancillari:[1,13,14],ani:[1,8,11,12,15],anoth:[1,8,11,12],anymor:1,api:2,appear:[1,8,11],append:[12,13,14,15,16],appli:[1,8,12],approach:4,appropi:[15,16],approxim:1,ar:[1,4,5,6,7,8,10,11,12,14,15,16],arbitrari:1,architectur:4,archiveprefix:4,arg:[1,7],argument:[1,7,8,15],aros:1,around:[10,16],arrai:11,arrow:8,arxiv:4,assert:[4,7,11,12,13,14],assum:8,attach:[1,5,8,11],attribut:[1,15],author:4,auto_stack:[1,10,13,14,15,16],auto_unbind:[1,8,10,13,15,16],automat:[1,8,11,12,13],auxiliari:8,avail:[4,10,11],avoid:[1,13],awar:[1,12],ax:[1,5,7,8,11,12],axes_nam:[1,4,5,8,11,12,13,14,15],axi:[0,3,5,8,11,15,16],axis1:1,axis2:1,axis_0:1,axis_1:1,axis_2:1,axis_n:1,b:[3,8,12],b_:8,backpropag:[10,16],backward:[1,4,10,12,16],base:[1,7,8,12],basic:[1,4,11,12],batch:[1,3,4,7,8,10,11,12,14,15],batch_1:[1,15],batch_m:1,batch_n:15,batch_siz:[7,10,16],becaus:[1,13,16],becom:[1,4,8],been:[1,8,11,15],befor:[1,10,11,15,16],beforehand:11,begin:[3,10,11,16],behav:[1,13],behaviour:[1,11,13,14],being:[1,3,7,8,12],belong:[1,5,10,11,12],besid:[1,7,8,11],best:4,better:11,between:[1,3,4,7,8,11,13,14,16],big:12,binom:3,blank:[1,5],bmatrix:3,bodi:11,bond:7,bond_dim:[4,7,10,16],bool:[1,5,7],border:7,both:[1,7,8,11,12,13,14,15],bottom:[7,16],bound:[1,7,8],boundari:7,bracket:1,bring:10,build:[1,4,9,12],built:[4,6,10,11,12,15],bunch:[1,12,15],c:[4,6,10,11],c_:8,cach:[1,13],calcul:13,call:[1,8,10,12,15,16],callabl:8,can:[1,4,6,7,8,10,11,12,13,14,15,16],cannot:[1,5,7,8,12,14],canon:[7,10,16],canonic:[7,10,16],canonicalize_univoc:7,carri:[1,8,11],cast:1,cat:16,cdot:[3,7],center:7,central:[1,11,16],certain:[1,4,5,8],chang:[1,4,11,13,15,16],change_s:1,channel:7,charact:1,check:[1,8,10,11,12,13,14,16],check_first:8,child:1,chose:14,classif:[7,10],classifi:[10,15],clone:[4,6],cnn:16,cnn_snake:16,cnn_snakesb:16,co:3,code:[1,4,11,16],coincid:[8,11],collect:[1,8,12,13],column:7,com:[4,6],combin:[1,11,15,16],come:[1,8,10,11,14,15],comma:8,command:[4,6],common:[1,10,15],compar:13,compat:[4,6,10,11],compil:[4,6,10,11],complet:[1,8],compon:[0,3,4,8,10,12],compos:[10,16],comput:[1,4,6,7,8,10,12,13,15,16],concept:4,condit:7,configur:4,conform:1,connect:[0,1,4,7,11,12,13,14,15],connect_stack:0,consecut:7,consist:[1,8],construct:[1,4,12,13],contain:[1,5,7,11,12,13],continu:10,contract:[1,4,7,8,9,10,11,13,14,15],contract_:[0,1,12],contract_between:[0,1,12],contract_between_:[0,1,12],contract_edg:[0,1,12],contract_edges_ip:[1,8],contrat:7,control:[1,11,13],conv2d:[7,16],conv_mp:7,conv_mps_lay:7,conv_pep:7,conv_tre:7,conveni:14,convent:[1,8,12,15],convmp:0,convmpslay:[0,16],convolut:[7,15,16],convpep:0,convtre:0,convump:0,convumpslay:0,convupep:0,convutre:0,copi:[0,1],core:[11,16],corner:7,correct:[1,13],correctli:8,correspond:[1,7,8,11,12,13,15],costli:[1,13],could:[7,10,11,13,15],count:10,coupl:[1,11,14],cours:[13,15],cpu:[1,10,15,16],creat:[1,4,5,7,8,9,12,13,14,15],creation:[4,8],crop:1,crossentropyloss:[10,16],cuda:[1,10,15,16],cum:[1,7,8],cum_percentag:[1,7,8,10,16],current:1,custom:[1,4,9],cut:[10,16],cutoff:[1,7,8],d:[3,4],dagger:[1,8],dangl:[1,8,10,11,12],data:[1,3,4,7,8,12,14,15,16],data_0:[1,14],data_1:1,data_nod:[1,12,14,15],data_node_:12,dataload:[10,16],dataset:[10,16],david:[3,4],ddot:3,de:[1,5,12],decompos:8,decomposit:[1,7,8,10,12,16],decreas:1,dedic:7,deep:[4,10],deepcopi:1,def:[1,10,15,16],defin:[1,3,4,7,11,12,14,15,16],degre:3,del:[1,8],delet:[1,8],delete_nod:1,depend:1,descent:[11,12],describ:[1,7,15],desir:[3,12,13,14,15],detail:[4,12],determin:4,develop:4,deviat:5,devic:[1,5,10,15,16],devot:13,df:1,diagon:[1,5,8],diagram:11,dict:1,dictionari:1,did:16,differ:[1,4,7,8,9,10,11,12,13,15,16],differenti:[4,9,11],dilat:7,dim:[3,10,16],dimens:[1,3,7,8,10,11,15],dimension:[11,16],directli:[1,4,6,8,11,12,13,14],disconnect:[0,1,11],distinct:[1,5],distinguish:14,distribut:5,divers:4,doe:[1,11,13],don:8,done:[1,11],down:7,down_bord:7,download:[4,6,16],drawn:5,drop_last:[10,16],dtype:1,dure:[1,10,13],e:[1,3,4,7,8,14],each:[1,3,5,7,8,10,11,12,13,14,15,16],earlier:11,easi:[1,4],easier:11,easili:[1,11,12,15,16],edg:[0,4,5,7,10,11,12,13,14,15,16],edge1:[1,8],edge2:[1,8],edgea:1,edgeb:1,edges_dict:1,effect:[1,12],effici:[1,4,11],einstein:12,einsum:[0,12],either:8,element:[1,4,5,7,8,11],element_s:[10,16],els:[10,15,16],emb:[10,15],emb_a:3,emb_b:3,embed:[0,10,15,16],embedd:3,empti:[0,1,11,12,13],enabl:[1,4,11,12,13],end:[1,3,8],engin:11,enhanc:4,enough:[1,7],ensur:[1,4],entail:[1,13],entangl:7,entir:1,enumer:1,environ:7,epoch:[1,10,15,16],epoch_num:[10,16],eprint:4,equal:[1,7,15],equilibrium:10,equival:[1,3,7,11,16],essenti:[1,4],etc:[1,8,11,15],evalu:12,even:[1,4,11,13,15],everi:[1,8,11,13,16],exact:1,exactli:1,exampl:[1,3,7,8,10,12,14,15],exceed:7,excel:4,except:5,exclud:[1,14],execut:8,exist:[1,8,12],expand:[1,3,15],experi:[1,4,11,13],experiment:4,explain:[1,11,12,13],explan:[1,8],explicitli:[1,14,15],explor:[4,11],express:8,extend:1,extens:[4,6,10,11],extra:[7,8],extract:11,extractor:16,extrem:7,ey:15,eye_tensor:15,f:[10,11,12,13,14,16],facilit:[1,4,8,11],fact:[1,14,15],fals:[1,5,7,8,10,12,13,15,16],familiar:[4,11],fashion:12,fashionmnist:16,faster:[1,15],fastest:4,featur:[1,3,8,11,12,13,14,15,16],feature_dim:[1,15],feature_s:7,fed:16,few:4,file:4,fill:[1,5,11,12,15],finish:[11,15],first:[1,3,4,7,8,9,11,12,13,14,16],fix:[1,11,12],flat:7,flatten:7,flavour:1,flip:16,folder:[4,6],follow:[1,4,6,7,8],forget:1,form:[1,7,10,11,12,14,16],former:[1,8],forward:[1,7,15,16],four:15,frac:[1,3,7,8],framework:4,friendli:4,from:[1,4,5,6,7,8,10,11,12,14,15,16],from_sid:7,from_siz:7,front:1,full:10,fulli:4,func1:8,func2:8,func:1,functool:16,fundament:4,further:[1,4],furthermor:[1,13,15],futur:[1,13],g:[1,7,8,14],gain:4,garc:4,gave:1,ge:[1,7,8],gener:[1,13],get:[1,10,11,15,16],get_axi:[1,11],get_axis_num:[1,11],get_edg:1,git:[4,6],github:[4,6],give:[1,11,13,14],given:[1,3,7],glimps:10,go:[1,8,10,12,13,16],goal:4,good:[1,10],gpu:[10,15],grad:[1,4,12],gradient:[1,4,11,12],graph:[1,11,13,15],grasp:4,greater:[1,15],greatli:4,grid:7,grid_env:7,group:8,guid:4,ha:[1,4,6,7,8,10,11,13,14,15,16],had:[1,8,13],hand:[1,4,15],happen:[1,13,16],har:4,hat:3,have:[1,4,6,7,8,10,11,12,13,14,15,16],heavi:15,height:[7,10,15],help:[1,13],henc:[1,5,8,10,11,12,13,15],here:[1,7,12],hidden:[1,14],high:[1,5],highli:4,hint:1,hold:[12,14],hood:13,horizont:7,how:[1,4,8,9,10,12,14,16],howev:[1,4,8,11,12,13],http:[4,6],hundread:12,hybrid:[4,9],i:[1,4,7,8,11,12,13,14,15],i_1:5,i_n:5,ideal:4,identif:4,idx:11,ijb:[8,12],ijk:8,im:8,imag:[7,10,15,16],image_s:[10,15,16],immers:4,implement:[1,4,10,16],improv:10,in_channel:[7,10,16],in_dim:[4,7,10],in_which_axi:1,includ:[1,4,7,11],incorpor:4,inde:1,independ:[1,8],index:[1,11,13],indic:[1,5,7,8,11,12,13,14],infer:[1,8,11,13,15],inform:[1,4,8,11,12,13],ingredi:[1,11],inherit:[1,8,14],inic:15,init:12,init_method:[1,11,12],initi:[0,1,7,10,11,12,15,16],inlin:7,inline_input:7,inline_mat:7,input:[1,7,8,10,11,12,13,14,15,16],input_edg:[1,14,15],input_nod:15,insid:[1,4,6],insight:12,instal:[2,10,11],instanc:[1,8,13,15],instanti:[1,11,13,14,15],instati:11,instead:[1,12,13,14,15,16],integr:4,intend:1,interior:7,intermedi:[1,12,14,15],intric:4,introduct:4,invari:[1,7,13,14],inverse_memori:1,involv:[1,8,13],is_attached_to:1,is_avail:[10,15,16],is_batch:[1,11],is_dangl:[1,11],is_data:[1,14],is_leaf:[1,14],is_node1:1,is_result:[1,14],is_virtu:[1,14],isinst:[1,12],isn:[10,11],isometri:7,issu:1,item:[10,16],iter:[1,13],its:[1,4,5,7,8,11,13,15,16],itself:[1,15],j:[1,3,4,8],jkb:[8,12],jl:8,jo:4,joserapa98:[4,6],just:[1,4,7,11,13,14,15,16],k:[4,8],keep:[1,7,8,13],kei:[1,4,11],kept:[1,7,8],kernel:7,kernel_s:[7,16],kerstjen:4,keyword:[1,7],kib:[8,12],kind:[1,10],klm:8,know:[1,11,12,13],known:15,kwarg:[1,7],l2_reg:[10,16],l:8,label:[7,10,14,16],lambda:[10,16],larger:1,largest:1,last:[1,7,8,15,16],later:15,latter:1,layer:[1,4,7,10,15,16],ldot:5,lead:[1,14],leaf:[1,8,14,15],leaf_nod:[1,14],learn:[1,4,8,10,11,12,13,14,15,16],learn_rat:[10,16],learnabl:12,least:[1,8,10,13,16],leav:13,left1:12,left2:12,left:[1,4,7,8,11,12,13,14,15,16],left_bord:7,left_down_corn:7,left_env:7,left_nod:7,left_up_corn:7,leg:1,len:[8,15],length:[1,7],less:[10,16],let:[10,11,12,13,15,16],level:[1,11],leverag:11,li:[1,4,11],librari:[4,11,16],like:[0,1,4,7,10,11,12,13,14,15,16],limit:5,line:[4,7,16],linear:[4,15],link:1,list:[1,5,7,8,11,12,15],load:[10,15],load_state_dict:15,loader:[10,16],local:[3,7],locat:[1,7],logit:[10,16],longer:[1,13],loop:[10,15],lose:[10,16],loss:16,loss_fun:[10,16],lot:[1,10],low:[1,5],lower:[1,5,7,8],lr:[10,16],m:[1,4,6,8],machin:[4,10],made:[1,8],mai:[1,4],main:[1,11,12,13,15],mainli:1,maintain:1,make:[1,4,6,10,11,12,13,14],make_tensor:1,manag:[1,15],mandatori:[1,8],mani:[1,10,11,13,16],manipul:11,manual:[1,15],manual_se:[10,16],map:3,master:[4,6],match:[1,7,8],matric:[7,8,12,15],matrix:[1,3,4,7,8,10,11,13,15],mats_env:7,max:[10,16],max_bond:7,maximum:[3,7],maxpool2d:16,mayb:8,mb:[10,16],mean:[1,5],megabyt:[10,16],mehtod:11,member:1,memori:[1,4,9,10,11,14,15,16],mention:15,method:[1,11,12,13,14,15],middl:7,might:[1,7,13,14,15],mile:3,mimic:15,minuend:8,misc:4,miscellan:[10,16],mit:4,mnist:10,mode:[1,7,8,10,14,15,16],model:[0,1,4,9,11,12,13,14],modif:1,modifi:[1,12],modul:[1,4,6,7,10,11,12,15,16],modulelist:16,monomi:3,monturiol:4,mooth:4,more:[1,4,7,8,11,12,13,14],most:[1,4,11,14],move:[1,7,11],move_nam:1,move_to_network:[1,8],movec:8,mp:[0,4,10,11,12,13,14,15,16],mps_layer:7,mpslayer:[0,4,10,15],much:[1,13,14,15,16],mul:[0,12],multi:11,multipl:[11,13],multipli:8,must:[1,8,11,15],my_model:4,my_nod:[1,13],my_paramnod:1,my_stack:1,mybatch:1,n:[1,3,4,5,7,10,16],n_:[1,3],n_batch:7,n_col:7,n_epoch:15,n_featur:[4,7,10,15],n_param:[10,16],n_row:7,name:[1,4,5,8,10,11,12,13,15,16],name_:1,necessari:[1,8,10,11,15,16],need:[1,8,10,11,12,13,15],neighbour:[1,7,12],nelement:[10,16],net2:1,net:[1,4,8,11,12,13,14],netwok:10,network:[0,4,5,7,8,9,13,14,15],neural:[4,7,9,10],never:[1,8,14],nevertheless:4,new_edg:[1,8],new_edgea:1,new_edgeb:1,new_mp:15,new_tensor:13,next:[1,7,8,11,12,13,14],nn:[1,4,7,10,11,12,15,16],no_grad:[10,16],node1:[1,4,8,11,12,13,14],node1_ax:[1,8],node1_list:1,node1_lists_dict:1,node2:[1,4,8,11,12,13,14],node2_ax:[1,8],node3:[12,13],node:[0,4,5,7,9,10,13,15,16],node_0:1,node_1:1,node_:[11,12,13,14],node_left:[1,8],node_n:1,node_right:[1,8],nodea:[1,8],nodeb:[1,8],nodec:[1,8],nodes_list:8,nodes_nam:1,nodesa:8,nodesb:8,nodesc:8,non:[1,7,8,12,15],none:[1,4,5,7,8,12,15],norm:1,normal:[1,5],notabl:4,notat:11,note:[1,4,8,11,12,15],noth:[1,11,13],now:[1,11,12,13,15,16],nu_batch:7,num:1,num_batch:[10,16],num_batch_edg:[1,14,15],num_epoch:[10,16],num_epochs_canon:10,num_test:[10,16],num_train:[10,16],number:[1,7,8,10,12,15,16],nummber:1,o:4,obc:7,object:[1,11,16],obtain:10,oc:7,occur:[1,13],off:[10,13,16],offer:4,often:[1,13],onc:[1,4,8,13],one:[1,7,8,10,11,12,13,14,15,16],ones:[0,1,3,7,11,12,14,16],ones_lik:[10,16],onli:[1,4,8,11,12,13,14,15],open:7,oper:[0,1,7,11,14,15],operand:[1,11,12],opposit:7,opt_einsum:[4,8,12],optim:[1,4,16],option:[1,3,4,5,7,8,11,15],order:[1,7,8,11,12,13,14,15],orient:7,origin:[1,3,8,10,14,16],orthogon:7,ot:7,other:[1,7,8,11,12,13,14,15,16],other_left:11,other_nod:13,otherwis:[1,7,8,16],otim:3,our:[10,11,12,13,16],out:[1,8,15],out_channel:[7,16],out_dim:[4,7,10],out_posit:7,output:[1,3,7,8,10,15],output_nod:[7,15],outsid:[4,6],over:[1,12],overcom:1,overhead:13,overload:1,overrid:[1,7,15],override_edg:1,override_nod:1,overriden:[1,12,15],own:[1,8,13,14],ownership:1,p:[1,4,10,16],packag:[4,6],pad:[7,16],pair:[1,7,16],pairwis:7,paper:[3,4,16],parallel:[1,7],param:[1,8,10,16],param_nod:[1,4,5,12,14],paramedg:1,paramet:[1,3,5,7,8,10,12,15,16],parameter:[1,12],parametr:[1,10,12,15,16],paramnod:[0,5,8,13,14,15],paramnode1:12,paramnode2:12,paramnode3:12,paramnodea:1,paramnodeb:1,paramstacknod:[0,8,13,14],pareja2023tensorkrowch:4,pareja:4,parent:[1,15],parenthesi:1,part:[1,12,15],partial:16,particular:[1,15],pass:[7,10,15],patch:[7,15],pattern:16,pave:10,pbc:7,pep:[0,4,15],perform:[1,4,7,8,10,11,12,13,15,16],period:7,permut:[0,1,12,13],permute_:[0,1,12],phase:8,phi:3,phys_dim:7,physic:[7,11],pi:3,piec:[11,15],pip:[4,6,10,11],pipelin:[4,10],pixel:[7,10,15,16],place:[1,8,12,13,15],plai:[7,14],pleas:4,plug:7,point:[1,10],poli:[0,15],pool:16,posit:[7,15],possibl:[1,10,11,12,13],potenti:4,power:[3,10],poza:4,practic:[1,4],practition:10,pred:[10,16],predict:[10,16],present:16,previou:[1,7,12,13,14,15,16],previous:12,primari:4,principl:[1,14],print:[1,7,8,10,16],probabl:10,problem:[1,13],process:[1,4,10],product:[4,7,8,10,11,13,15],profici:4,project:7,properli:15,properti:[1,7,8,11],proport:[1,7,8],provid:[1,4,8,11],prune:16,pt:15,purpos:[1,11,13,14],put:[1,7,10,16],pytest:[4,6],python:[4,6,10],pytorch:[1,4,6,8,10,11,12,13,15,16],q:8,qr:[1,7,8,12],qr_:[0,1,7],quantiti:[1,7,8],quantum:11,quit:[4,16],r:[4,8],r_1:8,r_2:8,rais:1,ram:4,rand:[0,1],randn:[0,1,3,4,8,11,12,13,14,15],random:[8,12],random_ey:15,rang:[1,4,7,8,10,11,12,13,14,15,16],rank:[1,3,7,8,12],rapid:4,rather:[1,11,13,14],re:[1,8,10,16],reach:10,readi:15,realli:[1,14],reattach:1,reattach_edg:1,recal:[1,15],recogn:12,recommend:[1,4],reconnect:12,recov:1,reduc:[1,7,10,16],redund:[1,8],refer:[1,2,8,11,12,13,14,15],referenc:1,regard:[1,8,12],relat:11,relev:[1,11],reli:4,relu:[4,16],remov:[1,8],repeat:[1,8],repeatedli:13,replac:1,repositori:[4,6],repres:11,reproduc:[1,13],requir:[1,8],requires_grad:1,res1:12,res2:12,reserv:[1,8,13],reset:[1,8,13,14,15],reset_tensor_address:1,reshap:[8,13],resiz:[10,16],respect:[1,7,8,12,13],rest:[1,12],restrict:8,result2:8,result:[1,4,7,8,12,13,14,15],resultant_nod:[1,14],retriev:[1,11,13,14],rez:4,rid:1,right1:12,right2:12,right:[1,4,7,8,11,12,13,14,15,16],right_bord:7,right_down_corn:7,right_env:7,right_nod:7,right_up_corn:7,ring:12,rise:1,role:[1,7,14,15],row:7,rowch:4,rq:[1,8],rq_:[0,1,7],run:[4,6,7,10,16],running_acc:10,running_test_acc:[10,16],running_train_acc:[10,16],running_train_loss:[10,16],s:[1,3,4,5,7,8,10,11,12,13,14,15,16],s_1:3,s_i:[1,7,8],s_j:3,s_n:3,sai:[1,13],same:[1,7,8,11,12,13,16],sampler:[10,16],satisfi:[1,7,8],save:[1,4,9,12,15],scenario:4,schwab:3,score:[10,16],second:[1,8,13],secondli:15,section:[11,12,14,16],see:[1,4,7,8,14,16],seen:14,select:[1,8,11,12,15],self:[1,7,15,16],send:[10,15,16],sens:[10,14],sento:15,separ:8,sepcifi:1,sequenc:[1,5,7,8,12,15],sequenti:[4,15],serv:10,set:[1,8,13,14,15,16],set_data_nod:[1,7,14,15],set_param:1,set_tensor:[1,13],set_tensor_from:[1,13,14,15],sever:[1,8],shall:[1,15],shape:[1,3,4,5,7,8,10,11,12,13,14,15],shape_all_nodes_layer_1:7,shape_all_nodes_layer_n:7,shape_node_1_layer_1:7,shape_node_1_layer_n:7,shape_node_i1_layer_1:7,shape_node_in_layer_n:7,share:[1,7,8,11,13,14],share_tensor:1,should:[1,3,5,7,8,11,12,13,14,15],show:11,side:[1,7,8,16],similar:[1,8,11,12,14,15],simpl:[4,11,12],simpli:[1,11,13,15],simplifi:[1,4],simul:11,sin:3,sinc:[1,4,6,7,8,10,11,12,13,14,15,16],singl:[1,7,8,12,15],singular:[1,7,8,10,12,16],site:[7,11],sites_per_lay:7,situat:14,size:[1,3,5,7,8,11,12,14,15,16],skip:[1,13],slice:[1,8,13,14],slide:1,slower:[1,13],small:12,smaller:1,smooth:4,snake:[7,16],so:[1,4,6,8,10,11,13,15],softmax:16,some:[1,7,8,10,11,12,13,14,15],someth:11,sometim:1,sort:[1,11,13,14],sourc:[1,3,4,5,6,7,8,10,11],space:[1,5,10,15,16],special:[1,12],specif:[1,4,14],specifi:[1,8,11,12,14],split:[0,1,7,11,12],split_0:[1,8],split_1:[1,8],split_:[0,1,12],split_ip:[1,8],split_ip_0:[1,8],split_ip_1:[1,8],sqrt:3,squar:8,stack:[0,1,3,7,10,12,13,14,15],stack_0:1,stack_1:1,stack_data:[1,8,15],stack_data_memori:[1,14],stack_data_nod:12,stack_input:15,stack_nod:[1,8,12,13,14],stack_result:[12,15],stackabl:8,stacked_einsum:[0,12],stackedg:[0,8,12],stacknod:[0,8,12,14],stai:1,standard:5,start:[1,7,11,13,15,16],state:[1,4,7,10,11,13,15],state_dict:15,staticmethod:16,std:[1,5,7,15],step:[1,4,9,16],stick:1,still:[1,8,12],store:[1,7,8,11,14,15],stoudenmir:3,str:[1,5,8],straightforward:4,strength:4,stride:[7,16],string:[1,7,8],structur:[1,4,7,11,15,16],sub:[0,12],subclass:[1,4,9,11,14,16],subsequ:[1,13],subsetrandomsampl:[10,16],substitut:1,subtract:8,subtrahend:8,successor:[0,8],suffix:1,suitabl:4,sum:[1,7,8,10,12,16],sum_:[1,7,8],summat:12,support:4,sure:[4,6,10,11],sv:[1,8],svd:[1,7,8,10],svd_:[0,1,7,12],svdr:[1,7,8],svdr_:[0,1,7],system:[4,6,10,11],t:[8,10,11,14,15],t_:[5,8],tailor:4,take:[1,7,10,13,14,15],taken:1,task:[1,7,14],temporari:[1,14],tensor:[0,3,4,5,7,9,14,15],tensor_address:[1,7,13,14],tensorkrowch:[1,3,5,6,7,8,9,12,14,15,16],tensornetwork:[1,4,5,8,9,12,14,16],termin:[4,6],test:[4,6,10,16],test_set:[10,16],th:7,than:[1,10,11,15,16],thank:11,thei:[1,4,5,6,7,8,11,12,14],them:[1,7,8,10,11,12,13,15,16],themselv:8,therefor:[1,4,6,11,15],thi:[1,4,7,8,10,11,12,13,14,15,16],thing:[1,8,10,14,16],think:13,those:[1,13,15],though:[1,8,12,15],thought:1,three:[1,8],through:[1,4,7,8,10,15,16],thu:[1,4,7,8,11,15],time:[1,3,4,7,8,9,11,12,15],titl:4,tk:[1,3,4,7,8,10,11,12,13,14,15,16],tn:15,togeth:[1,4,6,13,14],tool:[4,11,12],top:[1,4,10,12,16],torch:[1,3,4,5,7,8,10,11,12,13,14,15,16],torchvis:[10,16],total:[1,7,8,10,16],total_num:[10,16],totensor:[10,16],tprod:[0,12],trace:[1,8,10,14,15,16],track:[1,13],train:[1,4,11,12,13,15,16],train_set:[10,16],trainabl:[1,11,14,15,16],transform:[10,12,16],translation:[1,7,13,14],transpos:[10,15,16],treat:1,tree:[0,15],tri:[1,8],triangular:8,trick:[13,16],tupl:[1,5,7,8],turn:[1,7,13],tutori:[1,2,10,11,12,13,14,15,16],two:[1,7,8,11,12,13],two_0:8,type:[1,3,4,5,7,8,9,11,12,13],typeerror:1,u:[1,5,8],ump:0,umpslay:0,unbind:[0,1,12,13,15],unbind_0:[1,8],unbind_result:12,unbinded_nod:[12,13],unbound:[1,8],under:[4,13],underli:13,underscor:[1,8],understand:[4,11],undesir:[1,14],unfold:7,uniform:[1,5,7,13,14,15],uniform_memori:15,uniform_nod:[13,14],uninform:10,uniqu:13,unit:0,unitari:8,univoc:7,unix:[4,6],unless:1,unset:1,unset_data_nod:1,unset_tensor:1,unsqueez:16,until:[1,8,10],up:[1,7,8,11],up_bord:7,updat:[10,16],upep:0,upper:[5,7,8],ur_1sr_2v:8,us:[1,3,4,5,7,8,10,11,12,13,14,15,16],usag:1,useless:10,user:[1,4],usual:[1,7,8,14,15],usv:8,util:[10,16],utre:0,v:[1,4,6,8],valid:1,valu:[1,7,8,10,12,15,16],valueerror:1,vanilla:[11,12,13],variant:[12,15],variat:1,variou:4,vdot:3,vector:[3,7,10,15,16],veri:[1,10,11,13,15],verifi:[1,8],versatil:4,version:[1,7,8,12],vertic:[1,7],via:[1,7,8,11,12,13,14],view:[10,15,16],virtual:[1,8,13,14,15],virtual_nod:[1,14],virtual_stack:[1,8,14],virtual_uniform:[1,7,14,15],virtual_uniform_mp:1,visit:1,wa:[1,15,16],wai:[1,10,11,15],want:[1,11,12,13,15],we:[1,10,11,12,13,14,15,16],weight_decai:[10,16],well:[1,7,8],were:[1,12,14,15],what:[1,11,13,15],when:[1,7,8,10,11,12,13,14,15],whenev:[1,8],where:[1,3,5,7,8,10,11,12,13,15],whether:[1,5,7,11,13],which:[1,3,5,7,8,10,11,12,13,14,15],whilst:7,who:4,whole:[1,7,10,12,13,14,15],whose:[1,7,8,13],why:13,wide:[4,15],width:[7,10,15],wise:[1,8],wish:15,without:[1,5,8],won:15,word:[1,11],work:[4,12,16],workflow:[1,15],worri:[1,13],would:[1,7,8,11,13,16],wouldn:14,wow:16,wrap:[1,11],written:[4,6],x:[3,4,7,10,15,16],x_1:3,x_j:3,x_n:3,y1:16,y2:16,y3:16,y4:16,y:16,yet:[11,15],you:[4,6,10,11,12,13,14,15],your:[4,6,10,11,15],your_nod:13,yourself:4,zero:[0,1,10,15,16],zero_grad:[10,16],zeros_lik:13},titles:["API Reference","Components","&lt;no title&gt;","Embeddings","TensorKrowch documentation","Initializers","Installation","Models","Operations","Tutorials","First Steps with TensorKrowch","Creating a Tensor Network in TensorKrowch","Contracting and Differentiating the Tensor Network","How to save Memory and Time with TensorKrowch (ADVANCED)","The different Types of Nodes (ADVANCED)","How to subclass TensorNetwork to build Custom Models","Creating a Hybrid Neural-Tensor Network Model"],titleterms:{"1":[10,11,12,13,14,15],"2":[10,11,12,13,14,15],"3":[10,11,12,13,15],"4":10,"5":10,"6":10,"7":10,"class":8,"function":10,"import":10,The:[14,15],abstractnod:1,add:8,add_on:3,advanc:[13,14],api:0,ar:13,axi:1,between:12,build:[11,15],choos:10,cite:4,compon:[1,11,15],connect:8,connect_stack:8,content:2,contract:12,contract_:8,contract_between:8,contract_between_:8,contract_edg:8,convmp:7,convmpslay:7,convpep:7,convtre:7,convump:7,convumpslay:7,convupep:7,convutre:7,copi:5,creat:[11,16],custom:15,data:10,differ:14,differenti:12,disconnect:8,distinguish:12,document:4,download:10,edg:[1,8],einsum:8,embed:3,empti:5,everyth:15,exampl:4,faster:13,first:[10,15],how:[11,13,15],hybrid:16,hyperparamet:10,initi:5,instal:[4,6],instanti:10,introduct:[10,11,12,13,14,15],librari:10,licens:4,like:8,loss:10,manag:13,matrix:12,memori:13,mode:13,model:[7,10,15,16],mp:7,mpslayer:7,mul:8,name:14,network:[1,10,11,12,16],neural:16,node:[1,8,11,12,14],ones:5,oper:[8,12,13],optim:10,our:15,paramnod:[1,12],paramstacknod:1,pep:7,permut:8,permute_:8,poli:3,product:12,prune:10,put:15,qr_:8,rand:5,randn:5,refer:0,requir:4,reserv:14,rq_:8,run:13,save:13,set:10,setup:[10,11],skipp:13,split:8,split_:8,stack:8,stacked_einsum:8,stackedg:1,stacknod:1,start:10,state:12,step:[10,11,12,13,14,15],store:13,sub:8,subclass:15,successor:1,svd_:8,svdr_:8,tensor:[1,8,10,11,12,13,16],tensorkrowch:[4,10,11,13],tensornetwork:[11,13,15],time:13,togeth:15,tprod:8,train:10,tree:7,tutori:[4,9],type:14,ump:7,umpslay:7,unbind:8,unit:3,upep:7,utre:7,zero:5}})