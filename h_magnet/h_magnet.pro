
mm = 1.e-3;
deg = Pi/180.;
DefineConstant[
  Flag_FullMenu = {0, Choices{0,1}, Name "Parameters/Show all parameters"}
];

  DefineConstant[
    MUR_frame = {1000.0,
      Name Sprintf("Parameters/Frame/3Mu relative []")},
    BR_frame = {0.0,
      Name Sprintf("Parameters/Frame/3Br [T]")}
    MUR_magnet = {1.0,
      Name Sprintf("Parameters/Magnet/3Mu relative []")},
    BR_magnet = {1.0,
      Name Sprintf("Parameters/Magnet/3Br [T]")}
  ];

DefineConstant[
  // preset all getdp options and make them (in)visible
  R_ = {"MagSta_a", Name "GetDP/1ResolutionChoices", Visible 1,
	Choices {"MagSta_a", "MagSta_phi"}},
  C_ = {"-solve -v 5 -v2 -bin", Name "GetDP/9ComputeCommand", Visible 0}
  P_ = {"", Name "GetDP/2PostOperationChoices", Visible 0}
];

Group{
  // Geometrical regions (give litteral labels to geometrical region numbers)
  AirBox  = Region[2];
  Outer   = Region[3];

  Magnet = Region[0];
  Frame = Region[1];

  // Abstract Groups (group geometrical regions into formulation relevant groups)
  Vol_Air = Region[ {AirBox} ];
  Vol_Magnet = Region[ {Magnet, Frame} ];

  Vol_mu = Region[ {Vol_Air, Vol_Magnet}];

  Sur_Dirichlet_phi = Region[ Outer ];
  Sur_Dirichlet_a   = Region[ Outer ];

  Dom_Hgrad_phi = Region[ {Vol_Air, Vol_Magnet, Sur_Dirichlet_phi} ];
  Dom_Hcurl_a = Region[ {Vol_Air, Vol_Magnet, Sur_Dirichlet_a} ];
}

Function{
  mu0 = 4*Pi*1e-7;
  mu[ Vol_Air ] = mu0;

  br[Magnet] = Vector[0, 0, BR_magnet];
  hc[Magnet] = Vector[0, 0, BR_magnet/mu0];
  mu[Magnet] = mu0*MUR_magnet;
  br[Frame] = Vector[0, 0, BR_frame];
  hc[Frame] = Vector[0, 0, BR_frame/mu0];
  mu[Frame] = mu0*MUR_frame;

  nu[] = 1.0/mu[];
}

Jacobian {
  { Name Vol ;
    Case {
      { Region All ; Jacobian Vol ; }
    }
  }
}

Integration {
  { Name Int ;
    Case {
      { Type Gauss ;
        Case {
	  { GeoElement Triangle    ; NumberOfPoints 4 ; }
	  { GeoElement Quadrangle  ; NumberOfPoints 4 ; }
          { GeoElement Tetrahedron ; NumberOfPoints 4 ; }
	  { GeoElement Hexahedron  ; NumberOfPoints  6 ; }
	  { GeoElement Prism       ; NumberOfPoints  6 ; }
	}
      }
    }
  }
}

Constraint {
  { Name phi ;
    Case {
      { Region Sur_Dirichlet_phi ; Value 0. ; }
    }
  }
  { Name a ;
    Case {
      { Region Sur_Dirichlet_a ; Value 0. ; }
    }
  }
  { Name GaugeCondition_a ; Type Assign ;
    Case {
      { Region Dom_Hcurl_a ; SubRegion Sur_Dirichlet_a ; Value 0. ; }
    }
  }
}

FunctionSpace {
  { Name Hgrad_phi ; Type Form0 ; // magnetic scalar potential
    BasisFunction {
      { Name sn ; NameOfCoef phin ; Function BF_Node ;
        Support Dom_Hgrad_phi ; Entity NodesOf[ All ] ; }
    }
    Constraint {
      { NameOfCoef phin ; EntityType NodesOf ; NameOfConstraint phi ; }
    }
  }
  { Name Hcurl_a; Type Form1; // magnetic vector potential
    BasisFunction {
      { Name se;  NameOfCoef ae;  Function BF_Edge;
	Support Dom_Hcurl_a ;Entity EdgesOf[ All ]; }
    }
    Constraint {
      { NameOfCoef ae;  EntityType EdgesOf ; NameOfConstraint a; }
      { NameOfCoef ae;  EntityType EdgesOfTreeIn ; EntitySubType StartingOn ;
        NameOfConstraint GaugeCondition_a ; }
    }
  }
}

Formulation {
  { Name MagSta_phi ; Type FemEquation ;
    Quantity {
      { Name phi ; Type Local ; NameOfSpace Hgrad_phi ; }
    }
    Equation {
      Integral { [-mu[]*Dof{d phi} , {d phi} ] ;
        In Vol_mu ; Jacobian Vol ; Integration Int ; }
      Integral { [-mu[]*hc[] , {d phi} ] ;
        In Vol_Magnet ; Jacobian Vol ; Integration Int ; }
    }
  }
  { Name MagSta_a; Type FemEquation ;
    Quantity {
      { Name a  ; Type Local  ; NameOfSpace Hcurl_a ; }
    }
    Equation {
      Integral { [ nu[] * Dof{d a} , {d a} ] ;
        In Vol_mu ; Jacobian Vol ; Integration Int ; }
      Integral { [ -1/mu0 * br[] , {d a} ] ;
        In Vol_Magnet ; Jacobian Vol ; Integration Int ; }
    }
  }
}

Resolution {
  { Name MagSta_phi ;
    System {
      { Name A ; NameOfFormulation MagSta_phi ; }
    }
    Operation {
      Generate[A] ; Solve[A] ; SaveSolution[A] ;
      PostOperation[MagSta_phi] ;
    }
  }
  { Name MagSta_a ;
    System {
      { Name A ; NameOfFormulation MagSta_a ; }
    }
    Operation {
      Generate[A] ; Solve[A] ; SaveSolution[A] ;
      PostOperation[MagSta_a] ;
    }
  }
}

PostProcessing {
  { Name MagSta_phi ; NameOfFormulation MagSta_phi ;
    Quantity {
      { Name dphi   ;
	Value { Local { [ {d phi} ] ; In Dom_Hgrad_phi ; Jacobian Vol ; } } }
      { Name b   ;
	Value { Local { [ - mu[] * {d phi} ] ; In Dom_Hgrad_phi ; Jacobian Vol ; }
	        Local { [ - mu[] * hc[] ]           ; In Vol_Magnet ; Jacobian Vol ; } } }
      { Name h   ;
	Value { Local { [ - {d phi} ]        ; In Dom_Hgrad_phi ; Jacobian Vol ; } } }
      { Name hc  ; Value { Local { [ hc[] ]  ; In Vol_Magnet ; Jacobian Vol ; } } }
      { Name phi ; Value { Local { [ {phi} ] ; In Dom_Hgrad_phi ; Jacobian Vol ; } } }
    }
  }
  { Name MagSta_a ; NameOfFormulation MagSta_a ;
    PostQuantity {
      { Name b ; Value { Local { [ {d a} ]; In Dom_Hcurl_a ; Jacobian Vol; } } }
      { Name a ; Value { Local { [ {a} ]; In Dom_Hcurl_a ; Jacobian Vol; } } }
      { Name br ; Value { Local { [ br[] ]; In Vol_Magnet ; Jacobian Vol; } } }
    }
  }
}

PostOperation {
  { Name MagSta_phi ; NameOfPostProcessing MagSta_phi;
    Operation {
      Print[ dphi, OnElementsOf Vol_Magnet, File "dphi.pos" ] ;
      Echo[ Str["l=PostProcessing.NbViews-1;",
		"View[l].ArrowSizeMax = 100;",
		"View[l].CenterGlyphs = 1;",
		"View[l].RangeType = 1;",
		"View[l].CustomMin = 0;",
		"View[l].CustomMax = 6;",        
		"View[l].VectorType = 2;" ] ,
        File "tmp.geo", LastTimeStepOnly] ;         
      Print[ phi, OnElementsOf Vol_Magnet, File "phi.pos" ] ;
      Echo[ Str["l=PostProcessing.NbViews-1;",
		"View[l].ArrowSizeMax = 100;",
		"View[l].CenterGlyphs = 1;",
		"View[l].RangeType = 1;",
		"View[l].CustomMin = 0;",
		"View[l].CustomMax = 6;",        
		"View[l].VectorType = 2;" ] ,
        File "tmp.geo", LastTimeStepOnly] ;        
      Print[ h, OnElementsOf Vol_mu, File "h.pos" ] ;
      Echo[ Str["l=PostProcessing.NbViews-1;",
		"View[l].ArrowSizeMax = 100;",
		"View[l].CenterGlyphs = 1;",
		"View[l].RangeType = 1;",
		"View[l].CustomMin = 0;",
		"View[l].CustomMax = 6;",        
		"View[l].VectorType = 2;" ] ,
        File "tmp.geo", LastTimeStepOnly] ;    
      Print[ b, OnElementsOf Vol_mu, File "b.pos" ] ;
      Echo[ Str["l=PostProcessing.NbViews-1;",
		"View[l].ArrowSizeMax = 100;",
		"View[l].CenterGlyphs = 1;",
		"View[l].RangeType = 1;",
		"View[l].CustomMin = 0;",
		"View[l].CustomMax = 6;",        
		"View[l].VectorType = 2;" ] ,
        File "tmp.geo", LastTimeStepOnly] ;
    }
  }
  { Name MagSta_a ; NameOfPostProcessing MagSta_a ;
    Operation {
      Print[ a,  OnElementsOf Vol_mu,  File "a.pos", File "b.msh"  ];
      Echo[ Str["l=PostProcessing.NbViews-1;",
		"View[l].ArrowSizeMax = 100;",
		"View[l].CenterGlyphs = 1;",
		"View[l].RangeType = 1;",
		"View[l].CustomMin = 0;",
		"View[l].CustomMax = 6;",
		"View[l].VectorType = 2;" ] ,
	    File "tmp.geo", LastTimeStepOnly] ;
      Print[ b,  OnElementsOf Vol_mu,  File "b.pos", File "b.msh"  ];
      Echo[ Str["l=PostProcessing.NbViews-1;",
		"View[l].ArrowSizeMax = 100;",
		"View[l].CenterGlyphs = 1;",
		"View[l].RangeType = 1;",
		"View[l].CustomMin = 0;",
		"View[l].CustomMax = 6;",
		"View[l].VectorType = 2;" ] ,
	    File "tmp.geo", LastTimeStepOnly] ;
    }
  }
}
