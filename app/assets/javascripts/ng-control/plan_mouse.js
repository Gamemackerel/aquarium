function PlanMouse($scope,$http,$attrs,$cookies,$sce,$window) {

  function all_draggable(f) {
    aq.each($scope.plan.operations,f); 
    aq.each($scope.plan.modules,f);
    aq.each($scope.plan.current_module.input,f);    
    aq.each($scope.plan.current_module.output,f);        
  }  

  function current_draggable(f) {
    aq.each(aq.where($scope.plan.operations, op => op.parent_id == $scope.plan.current_module.id),f); 
    aq.each(aq.where($scope.plan.modules,     m =>  m.parent_id == $scope.plan.current_module.id),f);
    aq.each($scope.plan.current_module.input,f);    
    aq.each($scope.plan.current_module.output,f);        
  }  

  function draggable_in_multiselect(obj) {

    var m = $scope.multiselect;

    return  (( m.width >= 0 && m.x < obj.x && obj.x + obj.width < m.x+m.width ) ||
             ( m.width <  0 && m.x + m.width < obj.x && obj.x + obj.width < m.x )) &&
            (( m.height >= 0 && m.y < obj.y && obj.y + obj.height < m.y+m.height ) ||
             ( m.height <  0 && m.y + m.height < obj.y && obj.y + obj.height < m.y ));

  }  

  function snap(obj) {
    obj.x = Math.floor((obj.x+AQ.snap/2) / AQ.snap) * AQ.snap;
    obj.y = Math.floor((obj.y+AQ.snap/2) / AQ.snap) * AQ.snap;      
  }  

 $scope.multiselect = {};

 // Global mouse events ////////////////////////////////////////////////////////////////////////

 $scope.mouseDown = function(evt) {

    $scope.select(null);
    all_draggable(obj => obj.multiselect = false);

    $scope.multiselect = {
      x: evt.offsetX,
      y: evt.offsetY,
      width: 0,
      height: 0,
      active: true,
      dragging: false
    }      

  }

  $scope.mouseMove = function(evt) {

    if ( $scope.current_draggable && $scope.current_draggable.drag && !$scope.current_fv && !$scope.current_io ) {

      $scope.current_draggable.x = evt.offsetX - $scope.current_draggable.drag.localX;
      $scope.current_draggable.y = evt.offsetY - $scope.current_draggable.drag.localY;
      $scope.last_place = 0;

    } else if ( $scope.multiselect.dragging ) {

      current_draggable(obj => {
        if ( obj.multiselect ) {
          obj.x = evt.offsetX - obj.drag.localX;
          obj.y = evt.offsetY - obj.drag.localY;
        }
      });

    } else if ( $scope.multiselect.active ) {

      $scope.multiselect.width = evt.offsetX - $scope.multiselect.x;
      $scope.multiselect.height = evt.offsetY - $scope.multiselect.y;

    }

  }    

  $scope.mouseUp = function(evt) {

    if ( $scope.multiselect.dragging ) {
      $scope.multiselect.dragging = false;
    } else if ( $scope.multiselect.active  ) {
      current_draggable(obj => {
        if ( draggable_in_multiselect(obj) ) {
          obj.multiselect = true;
        }
      });
      $scope.multiselect.active = false;        
    }
  }

  // Draggable Object Events ////////////////////////////////////////////////////////////////

  $scope.draggableMouseDown = function(evt,obj) {

    if ( obj.multiselect ) {

      current_draggable(obj => {
        if ( obj.multiselect ) {
          obj.drag = {
            localX: evt.offsetX - obj.x, 
            localY: evt.offsetY - obj.y
          }
        }
      });

      $scope.multiselect.dragging = true;

    } else {

      $scope.select(obj); 
      all_draggable(d=>d.multiselect=false);
      $scope.current_draggable.drag = {
        localX: evt.offsetX - obj.x, 
        localY: evt.offsetY - obj.y
      };   

    }

    evt.stopImmediatePropagation();

  }

  $scope.draggableMouseUp = function(evt,obj) {

    if ( obj.multiselect ) {
      current_draggable(obj => snap(obj));
      delete obj.drag;
    } else {
      snap(obj);
      delete obj.drag;
    }        
    
  }

  $scope.draggableMouseMove = function(evt,obj) {}

  $scope.multiselect_x = function() {
    return $scope.multiselect.width > 0 ? $scope.multiselect.x : $scope.multiselect.x + $scope.multiselect.width;
  }

  $scope.multiselect_y = function() {
    return $scope.multiselect.height > 0 ? $scope.multiselect.y : $scope.multiselect.y + $scope.multiselect.height;
  }    

  $scope.multiselect_width = function() {
    return Math.abs($scope.multiselect.width);
  }

  $scope.multiselect_height = function() {
    return Math.abs($scope.multiselect.height);
  }  

  // Field Value Events ///////////////////////////////////////////////////////////////////////

  $scope.ioMouseDown = function(evt,obj,io) {

    all_draggable(d=>d.multiselect=false);

    if ( $scope.current_io && evt.shiftKey ) { // There is an io already selected, so make a wire
      $scope.connect($scope.current_io, $scope.current_draggable, io, obj);
    } else {
      $scope.select(obj);
      $scope.set_current_io(io,true);
    }

    if ( io.record_type == "ModuleIO" ) {
      console.log(io)
      console.log(["origin: ", $scope.plan.base_module.origin(io)]);
      console.log(["destinations: ", $scope.plan.base_module.destinations(io)]);
    }

    evt.stopImmediatePropagation();

  }    

  // Wire Events ////////////////////////////////////////////////////////////////////////////////

  $scope.wireMouseDown = function(evt, wire) {
    console.log(["mouse", wire])
    $scope.select(wire);
    evt.stopImmediatePropagation();
  }  

  // Keyboard

  $scope.select_all = function() {
    current_draggable(obj => obj.multiselect = true );
    $scope.select(null);    
  }

  $scope.delete = function() {

    if ( $scope.current_wire && $scope.current_wire.record_type == "Wire" ) {

      $scope.plan.remove_wire($scope.current_wire);
      $scope.current_wire = null;

    } else if ( $scope.current_wire && $scope.current_wire.record_type == "ModuleWire" ) {

      $scope.plan.current_module.remove_wire($scope.current_wire);
      $scope.current_wire = null;      

    } else if ( $scope.current_draggable && !$scope.current_fv ) {

      $scope.delete_object($scope.current_draggable);

    } else if ( $scope.multiselect ) {

      var objects = [];

      current_draggable(obj => {  
        if ( obj.multiselect ) {
          objects.push(obj);
        }
      });   

      aq.each(objects, obj => $scope.delete_object(obj));

      $scope.select(null)   

    }
  }      

  $scope.keyDown = function(evt) {

    switch(evt.key) {

      case "Backspace": 
      case "Delete":
        $scope.delete();
        break;

      case "Escape":
        $scope.select(null);
        all_draggable(op => op.multiselect = false)
        break;

      case "A":
      case "a":
        if (evt.ctrlKey) $scope.select_all()
        break

      case "M":
      case "m":
        if ( evt.ctrlKey ) $scope.plan.create_module($scope.current_op);
        break;

      case "O":
      case "o":
        if ( evt.ctrlKey ) $scope.plan.current_module.add_output();
        break;

      case "I":
      case "i":
        if ( evt.ctrlKey ) $scope.plan.current_module.add_input();
        break;

      default:

    }

  }      

}