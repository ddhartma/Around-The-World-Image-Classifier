//
var folder = document.getElementById("pathInput");

folder.onchange=function(){
  var files = folder.files,
      len = files.length,
      i;
      array = [];
  for(i=0;i<len;i+=1){
    console.log(folder)
    //console.log(files[i]);
    //array.push(files[i].webkitRelativePath + '<br>');
    array.push(files[i].webkitRelativePath);
    folder_path_original_images = array[0].split('/')[0];
  }
  folder_path_yolo_images = folder_path_original_images + '_atw/' + 'yolo';
  folder_path_personal_images = folder_path_original_images + '_atw/' + 'personal';
  folder_path_backup_xlsx = folder_path_original_images + '_atw/' + 'backup';
  document.getElementById("folder_path_original_id").setAttribute('value', folder_path_original_images);
  document.getElementById("folder_path_yolo_id").setAttribute('value', folder_path_yolo_images);
  document.getElementById("folder_path_personal_id").setAttribute('value', folder_path_personal_images);
  document.getElementById("folder_path_backup_id").setAttribute('value', folder_path_backup_xlsx);

}

function myFunction() {
  document.getElementById("demo").innerHTML = "Paragraph changed.";
}


