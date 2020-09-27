<?php
//This code shows how to Upload And Insert Image Into Mysql Database Using Php Html.
//connecting to uploadFile database.
$conn = mysqli_connect("localhost", "root", "", "uploadFile");
if($conn) {
//if connection has been established display connected.
echo "connected";
}
//if button with the name uploadfilesub has been clicked
if(isset($_POST['uploadfilesub'])) {
//declaring variables
$filename = $_FILES['uploadfile']['name'];
$filetmpname = $_FILES['uploadfile']['tmp_name'];
//folder where images will be uploaded
$folder = 'imagesuploadedf/';
//function for saving the uploaded images in a specific folder
move_uploaded_file($filetmpname, $folder.$filename);
//inserting image details (ie image name) in the database
$sql = "INSERT INTO `uploadedimage` (`imagename`)  VALUES ('$filename')";
$qry = mysqli_query($conn,  $sql);
if( $qry) {
echo "</br>image uploaded"; 
}
}

?>


<!DOCTYPE html>
<html>
<body>
<!--Make sure to put "enctype="multipart/form-data" inside form tag when uploading files -->
<form action="" method="post" enctype="multipart/form-data" >
<!--input tag for file types should have a "type" attribute with value "file"-->
<input type="file" name="uploadfile" />
<input type="submit" name="uploadfilesub" value="upload" />
</form>
</body>
</html>

