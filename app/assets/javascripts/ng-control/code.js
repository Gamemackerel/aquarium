(function() {

  var w = angular.module('aquarium'); 

  w.controller('codeCtrl', [ '$scope', '$http', '$attrs', '$cookies', 
                  function (  $scope,   $http,   $attrs,   $cookies ) {

    $scope.editor = null;

    $scope.aceLoaded = function(_editor) {
      window.dispatchEvent(new Event('resize'));       
      _editor.setShowPrintMargin(false);
      _editor.$blockScrolling = Infinity;  
      $scope.editor = _editor;
      $scope.editor.gotoLine(1); 
    };

    $scope.save = function(ot,name) {
      if ( !ot[name].no_edit ) {
        $http.post("/operation_types/code",{
          id: ot.id,
          name: name,
          content: ot[name].content
        }).then(function(response) {
          ot[name] = response.data;
          ot.recompute_getter('versions')
        });
      }
    }

  }]);

})();                    
