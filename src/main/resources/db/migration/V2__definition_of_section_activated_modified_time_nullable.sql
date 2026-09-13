-- The trafiken.nu getsections feed omits activatedModifiedTime for roughly half of all
-- sections (observed: ~47%), so it cannot be NOT NULL.
ALTER TABLE definition_of_section ALTER COLUMN activated_modified_time DROP NOT NULL;
