var originalUnit = preferences.rulerUnits;
preferences.rulerUnits = Units.PIXELS;

var paramList = [50];
// var paramList = [-20, -10, 10, 20];
// folder for contrast adjustment
// var targetDir = "/contrast_adjustment/";
// folder for contrast adjustment
var targetDir = "/test_images/";

var rootPath = "/g/code/papers/ImageHashing/data/COREL";
var rawFolder = new Folder(rootPath + "/query_database");
var fileList = rawFolder.getFiles("*");

for (var i = 0; i < fileList.length; i++) {
	for (var j = 0; j < paramList.length; j++){
		
		var fileName = fileList[i].name;
		var splits = fileName.split(".");
		var docRef = app.open(fileList[i]);
		// var savePath = rootPath + targetDir + splits[0] + "_" + j + "_" + paramList[j] + "." + splits[1];
		var savePath = rootPath + targetDir + splits[0] + "_contrast_" + paramList[j] + "." + splits[1];

		var artLayerRef = docRef.activeLayer;
		// adjust contrast
		artLayerRef.adjustBrightnessContrast(0, paramList[j]);
		// adjust brightness
		// artLayerRef.adjustBrightnessContrast(paramList[j], 0);
		
		var saveFile = new File(savePath);
		options = new TiffSaveOptions();
		docRef.saveAs(saveFile, options);
		docRef.close();
		docRef = null;
		artLayerRef = null;
		textItemRef = null;
	}
}

app.preferences.rulerUnits = originalUnit;