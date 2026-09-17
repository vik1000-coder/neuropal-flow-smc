% convert_connectome.m
% Script to convert the unreadable MATLAB table in 'connectome_syn_gj.mat' to a CSV file.

try
    disp('Loading connectome_syn_gj.mat...');
    data = load('connectome_syn_gj.mat');
    
    if isfield(data, 'conn_edges')
        disp('Found variable "conn_edges". Checking if it is a table...');
        if istable(data.conn_edges)
            disp('Yes, it is a table. Writing to CSV...');
            writetable(data.conn_edges, 'connectome_syn_gj.csv');
            disp('SUCCESS: Created connectome_syn_gj.csv');
        else
            disp('ERROR: "conn_edges" is not a table.');
            disp(class(data.conn_edges));
        end
    else
        disp('ERROR: Variable "conn_edges" not found in the MAT file.');
        disp('Variables found:');
        disp(fieldnames(data));
    end
catch ME
    disp('An error occurred:');
    disp(ME.message);
end
