let config_path () =
  Sys.getenv "HOME" ^ "/chukei.autoconfig"

let rec list_toml_files dir prefix =
  let entries = Sys.readdir dir in
  Array.fold_left (fun acc entry ->
    let full_path = Filename.concat dir entry in
    let key = if prefix = "" then entry else prefix ^ "/" ^ entry in
    if Sys.is_directory full_path then
      acc @ list_toml_files full_path key
    else if Filename.check_suffix entry ".toml" then
      key :: acc
    else
      acc
  ) [] entries

let get key =
  let path = Filename.concat (config_path ()) key in
  if Sys.file_exists path then
    let ic = open_in path in
    let n = in_channel_length ic in
    let s = really_input_string ic n in
    close_in ic;
    Some s
  else
    None

let set key value =
  let path = Filename.concat (config_path ()) key in
  let dir = Filename.dirname path in
  if not (Sys.file_exists dir) then
    ignore (Sys.command ("mkdir -p " ^ Filename.quote dir));
  let oc = open_out path in
  output_string oc value;
  close_out oc

let list_keys () =
  let path = config_path () in
  if Sys.file_exists path && Sys.is_directory path then
    list_toml_files path ""
  else
    []

let clear () =
  ()
